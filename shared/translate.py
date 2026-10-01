"""Hindi <-> English translation, on the device.

The whole answer pipeline (retrieval, grounding, citation checking) is English. To let a person ask in Hindi
and be answered in Hindi without rebuilding any of that, the question is translated to English, answered by
the same grounded pipeline, and the answer is translated back. Nothing is sent anywhere: two small Marian
models (Helsinki-NLP opus-mt hi-en and en-hi, converted once to CTranslate2 int8, ~78 MB each) run on the CPU
with `ctranslate2` and `sentencepiece`. No torch is needed at run time; see tools/build_translation_models.py
for the one-off conversion.

Machine translation of this size is rough, and the answer says so. What is protected on the way back are the
things a translation must never touch: evidence markers like [E1], numbers, dates, and identifiers such as
`devA-motor1` or `replace_bearing`. Those are masked before translation and restored afterwards.

Only Devanagari Hindi is recognised. Hindi typed in Latin letters ("kya hua") reads as English here.
"""
from __future__ import annotations

import os
import pathlib
import re
import threading

ROOT = pathlib.Path("models_cache") / "translate"
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
# things a translator would mangle: evidence markers, anything with a digit/underscore/hyphen/colon inside a
# token (ids, dates, numbers), and file-ish names
_KEEP = re.compile(r"\[E\d+\]|[A-Za-z0-9]+(?:[_\-:./][A-Za-z0-9]+)+|\b\d+(?:[.,]\d+)*\b|\b[A-Za-z]+\d+[A-Za-z0-9]*\b")


def detect(text: str) -> str:
    """'hi' if the text is mostly Devanagari, otherwise 'en'."""
    letters = _LETTER.findall(text or "")
    if not letters:
        return "en"
    deva = sum(1 for c in letters if _DEVANAGARI.match(c))
    return "hi" if deva / len(letters) >= 0.3 else "en"


class Translator:
    """One direction of one language pair. Loaded on first use, reused after that."""

    def __init__(self, pair: str, root: str | os.PathLike | None = None):
        self.pair = pair
        self.dir = pathlib.Path(root or ROOT) / pair
        self._tr = None
        self._src = None
        self._tgt = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        try:
            import ctranslate2, sentencepiece  # noqa: F401
        except Exception:
            return False
        return all((self.dir / f).exists() for f in ("model.bin", "source.spm", "target.spm"))

    def _load(self):
        if self._tr is None:
            with self._lock:
                if self._tr is None:
                    import ctranslate2
                    import sentencepiece as spm
                    self._src = spm.SentencePieceProcessor(model_file=str(self.dir / "source.spm"))
                    self._tgt = spm.SentencePieceProcessor(model_file=str(self.dir / "target.spm"))
                    self._tr = ctranslate2.Translator(str(self.dir), device="cpu", compute_type="int8",
                                                      inter_threads=1,
                                                      intra_threads=max(1, (os.cpu_count() or 4) - 2))
        return self._tr

    def _run(self, sentences: list[str]) -> list[str]:
        tr = self._load()
        batch = [self._src.encode(s, out_type=str) + ["</s>"] for s in sentences]
        res = tr.translate_batch(batch, beam_size=4, max_decoding_length=256, repetition_penalty=1.1)
        return [self._tgt.decode(r.hypotheses[0]).strip() for r in res]

    def translate(self, text: str) -> str:
        """Translate prose. Evidence markers, numbers and identifiers come back exactly as they went in."""
        out_lines = []
        for line in (text or "").split("\n"):
            if not line.strip():
                out_lines.append("")
                continue
            out_lines.append(" ".join(self._sentences(line)))
        return "\n".join(out_lines).strip()

    def _sentences(self, line: str) -> list[str]:
        """Protected tokens are cut out and never shown to the model. Masking them with a placeholder was
        tried first and failed: the model transliterated the placeholder ("X1X" came back as "एक्स1एक्स"), so
        identifiers and [E1] markers were lost. Translating only the prose BETWEEN them, and putting the
        originals back verbatim, cannot lose them. Word order around a token is the model's best guess."""
        pieces: list[tuple[bool, str]] = []            # (is_protected, text)
        pos = 0
        sentence = line.strip()
        for m in _KEEP.finditer(sentence):
            if m.start() > pos:
                pieces.append((False, sentence[pos:m.start()]))
            pieces.append((True, m.group(0)))
            pos = m.end()
        if pos < len(sentence):
            pieces.append((False, sentence[pos:]))
        prose = [i for i, (keep, s) in enumerate(pieces) if not keep and _LETTER.search(s)]
        done = self._run([pieces[i][1].strip(" ,:;()/-") for i in prose]) if prose else []
        out, k = [], 0
        for i, (keep, s) in enumerate(pieces):
            if keep:
                out.append(s)
            elif i in prose:
                out.append(done[k]); k += 1
            else:
                out.append(s.strip())                  # punctuation / spacing only
        return [" ".join(p for p in out if p).strip()]


_TRANSLATORS: dict[str, Translator] = {}


def _get(pair: str) -> Translator:
    if pair not in _TRANSLATORS:
        _TRANSLATORS[pair] = Translator(pair)
    return _TRANSLATORS[pair]


def available(lang: str = "hi") -> bool:
    return lang == "en" or (_get(f"{lang}-en").available and _get(f"en-{lang}").available)


def to_english(text: str, lang: str) -> str:
    if lang == "en":
        return text
    if lang == "hi" and _norm_hi(text) in _QUESTION_INDEX:
        return _QUESTION_INDEX[_norm_hi(text)]
    return _get(f"{lang}-en").translate(text)


# The device's own fixed sentences, written by hand. They are exact strings from Device.ask, so they never
# need machine translation, and machine translation would only make them worse.
FIXED_HI = {
    "Nothing on this device relates to that.": "इस डिवाइस पर इससे जुड़ी कोई जानकारी नहीं है।",
    "This device is offline, so it cannot look it up.": "यह डिवाइस ऑफ़लाइन है, इसलिए इसे खोज नहीं सकता।",
    "Turn the network back on and ask again, or teach it with “Teach it something”.":
        "नेटवर्क फिर से चालू करके दोबारा पूछें, या “Teach it something” से इसे सिखाएँ।",
    "Ask is set to search this device only. Settings → Ask will let it use the internet as well.":
        "Ask को केवल इसी डिवाइस में खोजने पर सेट किया गया है। Settings → Ask में इंटरनेट की अनुमति दी जा सकती है।",
    "No web search is configured on this device, so it cannot look it up. Set EDGE_SEARCH_PROVIDER, or teach it.":
        "इस डिवाइस पर वेब सर्च सेट नहीं है, इसलिए यह इसे खोज नहीं सकता। EDGE_SEARCH_PROVIDER सेट करें, या इसे सिखाएँ।",
    "I searched the internet too and found nothing that answers it.":
        "मैंने इंटरनेट पर भी खोजा, पर इसका जवाब नहीं मिला।",
    "I found nothing close enough to answer that. This device does hold records that touch on some of those "
    "words, but none of them answer the question. Try naming the part, code or symptom directly, or teach it.":
        "इसका जवाब देने लायक कुछ नहीं मिला। इस डिवाइस में इन शब्दों से मिलते-जुलते रिकॉर्ड हैं, पर उनमें से कोई भी "
        "सवाल का जवाब नहीं देता। पुर्ज़े, कोड या लक्षण का नाम सीधे लिखकर पूछें, या इसे सिखाएँ।",
    "From what this device holds:": "इस डिवाइस में मौजूद जानकारी से:",
    "From the internet:": "इंटरनेट से:",
    "From this device, and from the internet:": "इस डिवाइस और इंटरनेट से:",
}
_FIXED_RE = re.compile("(" + "|".join(re.escape(k) for k in sorted(FIXED_HI, key=len, reverse=True)) + ")")

_OUTCOME_HI = {"worked": "सफल", "failed": "असफल", "pending": "लंबित"}
_SRC_HI = {"confirmed by a technician": "तकनीशियन द्वारा पुष्टि",
           "suggested by physics, not yet confirmed": "भौतिकी का सुझाव, अभी पुष्टि नहीं"}
# The whole-record answers are templates filled with stored values (Device._overview, device._ep_line), so
# they are rewritten by pattern rather than machine-translated, which garbles them. (pattern, Hindi builder)
_TEMPLATES_HI = [
    (re.compile(r"(\d+) repair\(s\) on this machine have been recorded as having worked, each confirmed by "
                r"the machine's own sensor data afterwards\."),
     lambda m: f"इस मशीन पर {m[1]} मरम्मत सफल दर्ज की गई है, और हर एक की पुष्टि बाद में मशीन के अपने सेंसर डेटा से हुई।"),
    (re.compile(r"(\d+) thing\(s\) on this machine are still outstanding\."),
     lambda m: f"इस मशीन पर {m[1]} मामले अभी बाकी हैं।"),
    (re.compile(r"(\d+) episode\(s\) have been opened on this machine\."),
     lambda m: f"इस मशीन पर {m[1]} एपिसोड दर्ज हुए हैं।"),
    (re.compile(r"The (\d+) most recent:"), lambda m: f"सबसे हाल के {m[1]}:"),
    (re.compile(r"The most recent:"), lambda m: "सबसे हाल के:"),
    (re.compile(r"Episode #(\d+) on (\S+): (\S+) / (\S+) \((confirmed by a technician|suggested by physics, "
                r"not yet confirmed)\), (\d+) window\(s\), first seen ([^\s.]*)\."),
     lambda m: (f"एपिसोड #{m[1]}, {m[2]}: {m[3]} / {m[4]} ({_SRC_HI[m[5]]}), {m[6]} विंडो, "
                f"पहली बार {m[7]} को दिखा।")),
    (re.compile(r"Action taken: (\S+) \(outcome recorded as (\w+)\)\."),
     lambda m: f"की गई कार्रवाई: {m[1]} (नतीजा दर्ज: {_OUTCOME_HI.get(m[2], m[2])})।"),
    (re.compile(r"No action has been recorded against it yet\."),
     lambda m: "इस पर अब तक कोई कार्रवाई दर्ज नहीं है।"),
    (re.compile(r"The sensor then saw (\d+) consecutive healthy windows: symptom resolved\."),
     lambda m: f"इसके बाद सेंसर ने लगातार {m[1]} सामान्य विंडो देखीं: लक्षण ठीक हो गया।"),
    (re.compile(r"The sensor kept seeing the symptom afterwards: it did not hold\."),
     lambda m: "इसके बाद भी सेंसर को लक्षण दिखता रहा: समाधान टिका नहीं।"),
    (re.compile(r"Still verifying: (\d+) of (\d+) consecutive healthy windows so far\."),
     lambda m: f"अभी जाँच जारी है: अब तक {m[1]}/{m[2]} लगातार सामान्य विंडो।"),
    (re.compile(r"This is (\S+) at site (\S+), watched by device (\S+) on the (\S+) profile\. It holds "
                r"(\d+) episode\(s\) \((\d+) repair\(s\) recorded as worked, (\d+) as failed\), (\d+) thing\(s\) "
                r"you taught it, (\d+) item\(s\) shared from other devices and (\d+) photo\(s\)\. Everything "
                r"here was searched on this device, with no internet\."),
     lambda m: (f"यह {m[1]} है, साइट {m[2]} पर, डिवाइस {m[3]} द्वारा देखी जाती है ({m[4]} प्रोफ़ाइल)। इसमें {m[5]} "
                f"एपिसोड हैं ({m[6]} मरम्मत सफल दर्ज, {m[7]} असफल), आपके सिखाए {m[8]} तथ्य, दूसरे डिवाइसों से "
                f"साझा किए {m[9]} आइटम और {m[10]} फ़ोटो। यहाँ सब कुछ इसी डिवाइस पर खोजा गया, इंटरनेट के बिना।")),
]
for _k, _v in FIXED_HI.items():
    _TEMPLATES_HI.append((re.compile(re.escape(_k)), lambda m, _v=_v: _v))


def _segments_hi(text: str) -> list[tuple[bool, str]]:
    """Split an answer into (is_hand_written, text): spans a template matched become Hindi, the rest is left
    for the translator. Earlier, longer matches win where two overlap."""
    hits = []
    for rx, build in _TEMPLATES_HI:
        for m in rx.finditer(text):
            hits.append((m.start(), m.end(), build(m)))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    out, pos = [], 0
    for s, e, hindi in hits:
        if s < pos:
            continue
        if s > pos:
            out.append((False, text[pos:s]))
        out.append((True, hindi))
        pos = e
    if pos < len(text):
        out.append((False, text[pos:]))
    return out


def from_english(text: str, lang: str) -> str:
    """Translate an answer. Known device sentences use hand-written Hindi; whatever is left is machine
    translation, with [E1] markers, numbers and identifiers kept exactly as they were."""
    if lang == "en":
        return text
    if lang != "hi":
        return _get(f"en-{lang}").translate(text)
    out = []
    for done, piece in _segments_hi(text):
        if done or not _LETTER.search(piece):
            out.append(piece)
        else:
            lead = piece[:len(piece) - len(piece.lstrip())]
            out.append(lead + _get("en-hi").translate(piece))
    return re.sub(r"[ \t]{2,}", " ", "".join(out)).strip()


# How people actually phrase the four whole-record questions in Hindi. Matched after normalising spelling
# (chandrabindu, nukta, punctuation), because machine translation turns "ठीक किया गया" into "right here".
_QUESTIONS_HI = {
    "What problems have you seen recently?": (
        "मैंने कौन सी समस्याएं देखी हैं", "कौन सी समस्याएं देखी हैं", "क्या समस्याएं देखी हैं",
        "आपने कौन सी समस्याएं देखी हैं", "हाल में कौन सी समस्याएं आईं", "क्या खराबी देखी है"),
    "Has anything here been fixed before?": (
        "क्या यहां कुछ ठीक किया गया है", "क्या कुछ ठीक किया गया है", "क्या कुछ ठीक हुआ है",
        "क्या पहले कुछ ठीक किया गया है", "क्या कुछ सुधारा गया है", "क्या कोई मरम्मत हुई है"),
    "What do you know about me?": (
        "आप मेरे बारे में क्या जानते हैं", "तुम मेरे बारे में क्या जानते हो", "आपको क्या पता है",
        "आपकी मेमोरी में क्या है", "आपको क्या याद है"),
    "Is anything still unresolved?": (
        "क्या कुछ अब भी अनसुलझा है", "क्या कुछ अभी भी अनसुलझा है", "क्या कुछ अब भी बाकी है",
        "क्या कोई समस्या अभी भी बाकी है", "क्या कुछ अभी बाकी है"),
}


def _norm_hi(s: str) -> str:
    s = s.replace("ँ", "ं").replace("़", "")      # chandrabindu -> anusvara, drop nukta
    s = re.sub(r"[^ऀ-ॿ\s]", " ", s)
    s = re.sub(r"[।?!.,]", " ", s)
    return " ".join(s.split())


_QUESTION_INDEX = {_norm_hi(p): en for en, phrases in _QUESTIONS_HI.items() for p in phrases}
