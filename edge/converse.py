"""Conversation understanding: what KIND of message is this, and what is it ABOUT?

Everything Ask does after this point (retrieval, reading, answering) works on a plain, self-contained English question.
This module is what turns what a person actually typed into that question:

    raw message + the conversation so far
        -> casual talk?           (greeting, thanks, goodbye, "how are you")  -> a conversational reply, no retrieval
        -> request for a topic?   ("tell me about X", "can you explain X?")   -> the canonical question "What is X?"
        -> follow-up?             ("what are its inputs?", "give me an example") -> the question with the reference
                                     replaced by the current topic ("What are the inputs of <topic>?")
        -> unclear reference?     ("what does it do?" with two candidate topics, or none) -> ask which one
        -> otherwise              a standalone question, left exactly as asked

It works on the STRUCTURE of the language, never on a topic or a list of whole phrases: which words only frame a request
(tell, explain, overview, please ...), which are wh-words / auxiliaries / articles, which point back at something
(it, its, they, this, "the above"), and what is left over is the topic. No example subject appears in this file.

The "current topic" is the subject of the most recent standalone question in THIS conversation, so switching topics
switches the reference ("What is <A>? What is <B>? What does it do?" -> it = <B>) and an unrelated standalone question
never inherits the old topic. Topics are stored with each chat turn by Device.ask, so this module is pure: history in, decision out.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---- vocabulary: function words and framing words, never topics -----------------------------------------------------
_WH = frozenset("what whats what's who whom whose which when where why how".split())
_AUX = frozenset("is are was were am be been being do does did done can could will would should shall may might must has "
                 "have had having".split())
_FUNC = frozenset("a an the of in on at to for with by from as into onto about and or but if so than then also too very "
                  "just really there here some any all one ones its it's that's".split())
_REQUEST_CORE = frozenset("tell explain describe define give show teach discuss elaborate summarize summarise outline "
                          "overview introduction intro information info understand know learn help curious interested "
                          "wondering details facts basics explanation description definition idea".split())   # NOT "meaning": "the meaning of X" is the question
_REQUEST_EXTRA = frozenset("me us you your i i'd i'll i'm im my let lets let's please pls kindly want wanna need like love "
                           "hear see find out simple simply basic brief briefly short quick detailed full complete "
                           "little bit more general okay ok well hmm um uh sure thanks thank".split())
_FRAME = _WH | _AUX | _FUNC | _REQUEST_CORE | _REQUEST_EXTRA
# things that ask for MORE of the current topic rather than naming one
_ELLIPTICAL = frozenset("example examples instance instances sample another else more further next previous again "
                        "continue elaborate detail details overview summary explanation reason reasons".split())

_PERSONAL = frozenset("it its itself they them their theirs themselves he she him his her hers himself herself".split())
_PLURAL = frozenset("they them their theirs themselves these those".split())
_DEMO = frozenset("this that these those".split())
_ANAPHORA = re.compile(r"\b(?:the\s+above|the\s+previous(?:\s+one)?|the\s+same(?:\s+thing|\s+one)?|the\s+last\s+one"
                       r"|previous\s+answer|that\s+one|this\s+one)\b", re.I)

_TOKEN = re.compile(r"[\w'’]+(?:[-/.][\w]+)*", re.UNICODE)


def tokens(s: str) -> list[str]:
    return [t.replace("’", "'") for t in _TOKEN.findall(s or "")]


def _norm(t: str) -> str:
    return t.lower().strip("'.")


def topic_words(s: str) -> list[str]:
    """The words of `s` that carry the subject: everything except wh-words, auxiliaries, articles, prepositions, request
    framing and pronouns - in their original order and spelling."""
    return [t for t in tokens(s) if _norm(t) not in _FRAME and _norm(t) not in _PERSONAL and _norm(t) not in _DEMO]


# ---- casual conversation -------------------------------------------------------------------------------------------
_GREET = frozenset("hi hii hiii hello helloo hey heya hiya yo hola namaste namaskar howdy greetings sup wassup morning "
                   "afternoon evening".split())
_THANKS = frozenset("thanks thank thx ty thankyou cheers appreciated appreciate".split())
_BYE = frozenset("bye byee goodbye cya later farewell alvida".split())
_ACK = frozenset("ok okay cool nice great awesome alright fine good perfect sure fantastic wonderful".split())
_SOCIAL_PAD = frozenset("good there buddy friend team everyone all you u so much very a lot again too to for your the "
                        "assistant bot dear mate man sir madam how are is r doing going it whats what's up thanks "
                        "see take care night welcome youre you're no problem anytime sorry my today tonight everyone guys "
                        "folks".split())
_HINDI = re.compile(r"[ऀ-ॿ]")


def casual(msg: str) -> tuple[str, str] | None:
    """(kind, reply) if the WHOLE message is social talk and carries no topic; else None.

    Decided from the words, not from a list of sentences: every word must be a social/function word and at least one
    must anchor it (a greeting, thanks, goodbye, "how are you", an acknowledgement)."""
    t = [_norm(x) for x in tokens(msg)]
    low = " " + " ".join(t) + " "
    if _HINDI.search(msg or ""):
        for kind, rx in _HINDI_PATTERNS:
            if rx.search(msg):
                return kind, _REPLY[kind][1]
        return None
    if not t or len(t) > 9:
        return None
    for kind, rx in _ABOUT_ME:                  # identity / capability questions name something to answer
        if rx.match(low.strip()):
            return kind, _REPLY[kind][0]
    known = _SOCIAL_PAD | _GREET | _THANKS | _BYE | _ACK | _WH | _AUX | _REQUEST_CORE | _REQUEST_EXTRA | _FUNC
    if any(w not in known for w in t):
        return None
    has = lambda s: any(w in s for w in t)                                  # noqa: E731
    if has(_THANKS):
        return "thanks", _REPLY["thanks"][0]
    if re.search(r"\byou'?re welcome\b|\bwelcome\b|\bno problem\b|\banytime\b", low):
        return "welcome", _REPLY["welcome"][0]
    if has(_BYE) or re.search(r"\bsee you\b|\btake care\b|\bgood night\b", low):
        k = "goodnight" if "night" in t else "bye"
        return k, _REPLY[k][0]
    if re.search(r"\bhow\b.*\b(?:are|r)\b.*\b(?:you|u)\b|\bhow'?s it going\b|\bhow do you do\b", low):
        return "how_are_you", _REPLY["how_are_you"][0]
    if re.search(r"\bwhat'?s\s*up\b|\bwhats\s*up\b|\bsup\b|\bwassup\b", low):
        return "whats_up", _REPLY["whats_up"][0]
    if has(_GREET):
        part = next((w for w in t if w in ("morning", "afternoon", "evening")), None)
        return "greeting", (f"Good {part}! How can I help?" if part else _REPLY["greeting"][0])
    if has(_ACK):
        return "ack", _REPLY["ack"][0]
    return None


_ABOUT_ME = (
    ("identity", re.compile(r"^(?:who are you|who r u|what are you|what is your name|what'?s your name|tell me about yourself"
                            r"|introduce yourself)$")),
    ("help", re.compile(r"^(?:help|help me|what can you do|what do you do|how can you help(?: me)?|how do i use (?:this|you)"
                        r"|what can i ask(?: you)?|what can you help me with)$")),
)
_HINDI_PATTERNS = (
    ("thanks", re.compile(r"धन्यवाद|शुक्रिया")), ("bye", re.compile(r"अलविदा")),
    ("how_are_you", re.compile(r"कैसे\s+हो|आप\s+कैसे\s+हैं")),
    ("identity", re.compile(r"कौन\s+हैं|कौन\s+हो")), ("help", re.compile(r"मदद")),
    ("greeting", re.compile(r"नमस्ते|नमस्कार")),
)
# (English, Hindi). Written by hand; no model is involved in casual talk.
_REPLY = {
    "greeting": ("Hi! How can I help?", "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?"),
    "thanks": ("You're welcome!", "आपका स्वागत है!"),
    "welcome": ("Anytime! What would you like to look up next?", "कभी भी! आगे क्या खोजना चाहेंगे?"),
    "bye": ("Goodbye! Come back any time.", "अलविदा! कभी भी वापस आइए।"),
    "goodnight": ("Good night!", "शुभ रात्रि!"),
    "how_are_you": ("I'm doing well, thanks for asking. How can I help?",
                    "मैं ठीक हूँ, पूछने के लिए धन्यवाद। मैं कैसे मदद कर सकता हूँ?"),
    "whats_up": ("Not much - I'm here and ready. What would you like to know?",
                 "कुछ खास नहीं - मैं तैयार हूँ। आप क्या जानना चाहेंगे?"),
    "ack": ("Okay. Ask me anything when you're ready.", "ठीक है। जब तैयार हों, कुछ भी पूछिए।"),
    "identity": ("I'm Machine Memory, the assistant on this device. I remember what this machine has been through and what "
                 "fixed it, I search an offline library of facts and engineering concepts, and I only answer from sources "
                 "I can cite. When I don't know, I say so.",
                 "मैं Machine Memory हूँ, इस डिवाइस का सहायक। मुझे याद रहता है कि इस मशीन के साथ क्या हुआ और क्या ठीक हुआ; मैं "
                 "तथ्यों और इंजीनियरिंग अवधारणाओं की ऑफ़लाइन लाइब्रेरी में खोजता हूँ, और केवल उन्हीं स्रोतों से जवाब देता हूँ जिनका "
                 "हवाला दे सकूँ। जब मुझे पता नहीं होता, मैं बता देता हूँ।"),
    "help": ("You can ask about this machine's faults and repairs, about what other machines in the fleet found (once "
             "synced), or about facts and engineering concepts, for example \"What is a PLC?\" or \"Tell me about "
             "photosynthesis\". You can also follow up with \"What does it do?\". If I don't know something I'll say so.",
             "आप इस मशीन की खराबियों और मरम्मत के बारे में, फ्लीट की दूसरी मशीनों के निष्कर्षों के बारे में, या तथ्यों और "
             "इंजीनियरिंग अवधारणाओं के बारे में पूछ सकते हैं। जो मुझे नहीं पता, वह मैं बता दूँगा।"),
}


# ---- the result ----------------------------------------------------------------------------------------------------
@dataclass
class Understanding:
    kind: str                       # casual | clarify | knowledge | followup
    resolved: str = ""              # the self-contained question retrieval should use
    topics: list[str] = field(default_factory=list)   # the subject(s) of this turn, stored with it
    reply: str | None = None        # set for casual / clarify
    sub: str = ""                   # casual subtype
    notes: dict = field(default_factory=dict)         # for the trace


# ---- requests for a topic ("tell me about X") ----------------------------------------------------------------------
_TRAILING = re.compile(r"\s*(?:,?\s*please|,?\s*pls|\bin\s+(?:simple|plain|short|brief)\s+(?:terms|words|english)|\bfor\s+me"
                       r"|\bbriefly|\bin\s+detail|\bin\s+short)\W*$", re.I)


def canonical(msg: str) -> tuple[str, bool]:
    """Turn a request for information into the canonical question, generically.

    "Tell me about X", "Can you explain X?", "Give me an overview of X", "I'd like to know about X", "Help me understand
    X", "What can you tell me about X?" ... all carry the same thing: framing words + a topic. The framing is a closed class
    of function/request words; whatever follows them is the topic, and the question is "What is <topic>?". A request that
    already contains a real question after its frame ("explain how X works") or a relation ("inputs of X") keeps what it
    says, minus the politeness."""
    text = (msg or "").strip()
    toks = tokens(text)
    i, seen_request, embedded = 0, False, None
    article = ""
    while i < len(toks) and _norm(toks[i]) in _FRAME:
        n = _norm(toks[i])
        article = toks[i] if n in ("a", "an", "the") else ""
        if n in _REQUEST_CORE:
            seen_request = True
        elif seen_request and n in _WH:
            embedded = i                          # "explain WHAT a X does", "tell me HOW X works"
            break
        i += 1
    if embedded is not None:
        return _tidy_embedded(" ".join(toks[embedded:])), True
    if not seen_request or i >= len(toks):
        return _tidy_embedded(text), False
    rest = _TRAILING.sub("", " ".join(toks[i:]))
    rest_toks = tokens(rest)
    while rest_toks and _norm(rest_toks[-1]) in _FRAME:       # trailing "does", "is" after the topic
        rest_toks.pop()
    if not rest_toks or len(rest_toks) > 8:
        return text, False
    if re.search(r"\b(?:of|between|versus|vs)\b", " ".join(t.lower() for t in rest_toks)):
        return text, False                       # "give me the inputs of X": already says which aspect
    noun = " ".join(([article] if article else []) + rest_toks)
    return f"What {'are' if _plural_noun(rest_toks[-1]) else 'is'} {noun}?", True


def _plural_noun(w: str) -> bool:
    if len(w) > 2 and w[:-1].isupper() and w.endswith("s"):          # acronym plurals ("PLCs")
        return True
    w = w.lower()
    return len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is", "ics", "ous"))


def _tidy_embedded(text: str) -> str:
    """'explain what a X does' / 'how X works' -> well-formed questions the reader was trained on."""
    t = re.sub(r"^\s*(?:please\s+)?(?:can|could|would)\s+you\s+(?:please\s+)?(?:tell|explain|show)\s+(?:me\s+)?", "", text,
               flags=re.I).strip()
    t = re.sub(r"^(?:explain|describe|tell me)\s+(?=(?:what|how|why|where|when|who)\b)", "", t, flags=re.I)
    t = t.rstrip("?.! ")
    aux = r"(?!(?:does|do|did|is|are|was|were|can|could|will|would|should|has|have)\b)"
    m = re.match(rf"(?i)^what\s+{aux}(?:(?:a|an|the)\s+)?(.+?)\s+(does|do)$", t)
    if m:
        return f"What does {m.group(1)} do?"
    m = re.match(rf"(?i)^how\s+{aux}(?:(?:a|an|the)\s+)?(.+?)\s+works?$", t)
    if m:
        return f"How does {m.group(1)} work?"
    return text


# ---- references ----------------------------------------------------------------------------------------------------
def is_followup(msg: str) -> bool:
    """Does this message lean on the conversation instead of standing on its own?

    Three signals, any one is enough: a pronoun / anaphoric phrase that points back ("it", "their", "the above");
    a demonstrative with almost nothing of its own to say ("why is this important?"); or no topic at all - what is left
    after removing the framing is empty or only asks for more ("give me an example", "and then?", "why?")."""
    toks = [_norm(t) for t in tokens(msg)]
    if not toks:
        return False
    own = [t for t in topic_words(msg) if _norm(t) not in _ELLIPTICAL]
    if _PERSONAL.intersection(toks) or _ANAPHORA.search(msg or ""):
        return True
    if _DEMO.intersection(toks) and len(own) <= 1:
        return True
    return not own


_VERBISH = re.compile(r"(?i)^(?:work|works|working|happen|happens|found|used|called|made|located|mean|means|stand|stands"
                      r"|exist|exists|occur|occurs|wrote|written|write|writes|invented|invent|discovered|discover|painted"
                      r"|founded|built|build|created|create|composed|directed|born|died|named|different|differ|differs"
                      r"|better|worse|similar|difference)$")


def _topic_of(msg: str) -> str:
    """The subject phrase of a STANDALONE question. "<aspect> of <entity>" is about the entity ("inputs of a system" -> system,
    "capital of <place>" -> <place>); otherwise the topic words themselves, minus leftover verbs. Identifiers (plot 91, P0301)
    are kept."""
    m = re.search(r"(?i)\bof\s+(?!the\s*$)(.+)$", msg.rstrip(" ?.!"))
    scope = m.group(1) if m else msg
    words = topic_words(scope)
    keep = [w for w in words if not _VERBISH.match(w)]
    return " ".join(keep or words).strip(" ?.!,")


_COMPARE = re.compile(r"\b(?:vs\.?|versus)\b|\bdifference between\b|\bcompare\b|\bdiffer(?:s|ent)?\s+from\b"
                      r"|\b(?:better|worse|bigger|smaller|faster|slower)\s+than\b|\bsimilar\s+to\b", re.I)


def _split_compare(msg: str) -> list[str]:
    parts = re.split(r"(?i)\b(?:vs\.?|versus|and|or|compared (?:to|with)|difference between|differ(?:s|ent)?\s+from"
                     r"|(?:better|worse|bigger|smaller|faster|slower)\s+than|similar\s+to|from)\b", msg)
    out = [_topic_of(p) for p in parts]
    return [p for p in out if p]


def _substitute(msg: str, topic: str) -> str:
    """Replace the reference in `msg` with the topic. Only the FIRST pronoun points at the topic: in "how does it process
    them" the second one means something from the answer, and guessing it would be inventing."""
    t = _ANAPHORA.sub(topic, msg)
    topic_toks = {_norm(x) for x in tokens(topic)}
    # "this system" / "that process": the noun after the demonstrative IS the reference - drop the demonstrative
    t = re.sub(r"(?i)\b(?:this|that|these|those)\s+(?=(\w+))",
               lambda m: "" if _norm(m.group(1)) in topic_toks else m.group(0), t)
    poss = topic + ("'" if topic.endswith("s") else "'s")
    t, n = re.subn(r"(?i)\b(?:its|their|theirs)\b", poss, t, count=1)
    if not n:
        t = re.sub(r"(?i)\b(?:it|they|them|this|that|these|those|itself|themselves)\b", topic, t, count=1)
    return re.sub(r"\s+", " ", t).strip()


def understand(msg: str, history: list[dict]) -> Understanding:
    """Classify `msg` and resolve it against the conversation. `history` is the recent turns of THIS conversation, oldest
    first; each may carry the "topics" list that Device.ask stored with it."""
    text = (msg or "").strip()
    c = casual(text)
    if c:
        return Understanding("casual", reply=c[1], sub=c[0], notes={"why": "only social words"})

    follow = is_followup(text)
    prev = next((h for h in reversed(history) if h.get("topics")), None)
    notes: dict = {"followup": follow}

    if not follow:
        canon, is_request = canonical(text)
        topics = _split_compare(text) if _COMPARE.search(text) else []
        one = _topic_of(canon)
        topics = topics if len(topics) >= 2 else ([one] if one else [])
        return Understanding("knowledge", resolved=canon, topics=topics,
                             notes=notes | {"request_frame": is_request, "topic_source": "this question"})

    # ---- a follow-up: it needs something to refer to
    if not prev:
        return Understanding("clarify", reply="I'm not sure what you're referring to yet. Which topic do you mean? Ask me "
                                              "about it by name and I'll look it up.", notes=notes | {"why": "no topic yet"})
    ptopics = list(prev["topics"])
    toks = {_norm(t) for t in tokens(text)}
    plural = bool(_PLURAL.intersection(toks))
    refers = bool((_PERSONAL | _DEMO).intersection(toks)) or bool(_ANAPHORA.search(text))
    if len(ptopics) >= 2 and not plural and refers:
        return Understanding("clarify", reply=f"Do you mean {_join_or(ptopics)}?",
                             notes=notes | {"why": "two candidate topics", "candidates": ptopics})
    topic = " and ".join(ptopics) if len(ptopics) >= 2 else ptopics[0]
    if refers:
        resolved = _substitute(text, topic)
    else:
        core = text.rstrip("?.! ")                  # ellipsis ("give me an example", "why?"): the subject is the topic
        more = re.search(r"(?i)\b(example|examples|instance|sample|overview|summary|reason|reasons|detail|details"
                         r"|explanation)\b", core)
        resolved = f"{core} of {topic}" if more else f"{core} about {topic}"
    resolved, _ = canonical(resolved)
    return Understanding("followup", resolved=resolved, topics=ptopics,
                         notes=notes | {"topic_source": "previous turn", "referred_to": topic})


def _join_or(items: list[str]) -> str:
    items = [i if re.match(r"(?i)^(?:the|a|an)\b", i) else f"the {i}" for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]
