"""One edge device: the orchestration of store, journal, novelty gate, outcome verifier, policy and sync.

Point types in the local shard (all carry machine_id):
  baseline  healthy fingerprints of this machine (vib only). Used by the gate and the verifier.
  exemplar  up to MAX_EXEMPLARS fingerprints per episode (vib only). The gate merges against these.
  episode   one abnormal-state occurrence: fingerprint + note vectors + all lifecycle fields.
Fleet knowledge lives in a SEPARATE read-only mirror shard (Qdrant's documented dual-shard pattern).

Episode lifecycle:  open -> verifying (action recorded) -> closed (verdict final AND technician outcome set)
Every write goes journal (SQLite) -> shard (flushed) -> journal marked applied.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import math
import os
import pathlib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from edge import fleet_hint, local_detector, machine_card, policy, profiles, storage_os, vehicle_risk
from edge import physics as P
from edge import verifier as V
from edge.fingerprint import Baseline
from edge.gate import GateConfig, NoveltyGate, calibrate
from edge.mirror import Mirror
from edge.crypto import NoteCipher, load_or_create_key
from edge.outbox import Outbox
from edge.store_edge import EdgeStore, StorePoint, canonical_id
from shared import ids
from shared.embed import Embedder
from shared.redact import redact
from shared.schema import ActionCode, DamageMode, FaultClass, FollowUp, RootCause

MAX_EXEMPLARS = 30
HINT_WINDOWS = 10          # physics-hint votes collected from the first windows of an episode
TERMINAL = ("closed",)
RETENTION_EVERY_S = 3600.0
CLOCK_WARN_S = 120.0       # device vs cloud clock difference above which the device warns (and the cloud corrects)
FLEET_TEXT_MODEL = "BAAI/bge-small-en-v1.5"   # default until the cloud announces its model (/v1/mirror/head)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


class _NoteVault:
    """The device's view of its store with technician notes encrypted at rest (edge/crypto.py): every payload
    written carries an encrypted note_text, every payload read comes back decrypted. Everything else passes through."""

    def __init__(self, store: EdgeStore, cipher: NoteCipher):
        self._s, self._c = store, cipher

    def __getattr__(self, name):
        return getattr(self._s, name)

    def _enc(self, p: dict) -> dict:
        return p | {"note_text": self._c.encrypt(p["note_text"])} if p.get("note_text") else p

    def _dec(self, p: dict) -> dict:
        return p | {"note_text": self._c.decrypt(p["note_text"])} if p.get("note_text") else p

    def upsert(self, points, **kw):
        return self._s.upsert([StorePoint(p.id, self._enc(p.payload), p.vib, p.note, p.bm25_text, p.image)
                               for p in points], **kw)

    def modify(self, pid, fn):
        out = self._s.modify(pid, lambda cur: self._enc(dict(fn(self._dec(cur)))))
        return None if out is None else self._dec(out)

    def get(self, pid, **kw):
        r = self._s.get(pid, **kw)
        if r is not None:
            r.payload = self._dec(r.payload)
        return r

    def retrieve(self, ids_, **kw):
        out = self._s.retrieve(ids_, **kw)
        for r in out:
            r.payload = self._dec(r.payload)
        return out

    def scroll(self, **kw):
        for r in self._s.scroll(**kw):
            r.payload = self._dec(r.payload)
            yield r

    def search(self, **kw):
        out = self._s.search(**kw)
        for h in out:
            h.payload = self._dec(h.payload)
        return out


def _json_safe(x):
    """Round floats and drop NaN/inf so physics diagnoses can live in a payload."""
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, float):
        return round(x, 4) if np.isfinite(x) else None
    return x


class VersionConflict(RuntimeError):
    """The episode changed since the caller read it (another technician or tab). Nothing was written."""

    def __init__(self, eid: str, expected: int, current: int):
        super().__init__(f"CONFLICT: episode {eid[:8]} is at version {current}, you edited version {expected}; "
                         "reload it and re-apply your change")
        self.current = current


_STOPWORDS = frozenset("""
about after again all also and any are because been before being between both but can cant come could did
does doing done down each even every for from get gets got had has have having here how into its just like
make many may more most much must not now off once only other our out over own same she should since some
such than that the their them then there these they thing things this those through too under until use
used using very was way were what when where which while who why will with would you your yours
tell explain describe define give show teach discuss elaborate summarize summarise outline overview introduction information
info details explanation description want wanna need know learn understand help please kindly briefly simple basic quick short
""".split())


def _content_words(s: str) -> set[str]:
    """Words that carry meaning, for deciding whether a question actually matches anything in memory."""
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower())
            if len(w) > 2 and w not in _STOPWORDS}


def _stem(w: str) -> str:
    """Plural-insensitive: "states" must match "state". Deliberately nothing smarter - a real stemmer would also
    merge words that mean different things."""
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _covered(qw: set[str], text: str, plural: bool = True) -> set[str]:
    """Which of the question's content words does this text contain? Compound-aware: "tamilnadu" is covered by
    "Tamil Nadu", because people type place names both ways and the text only ever holds one of them. `plural`
    also lets "states" meet "state"; it is off for what people typed in, where one loose word is already enough."""
    f = _stem if plural else (lambda w: w)
    toks = [t for t in re.findall(r"[a-z0-9]+", (text or "").lower())]
    have = {f(t) for t in toks}
    have |= {f(a + b) for a, b in zip(toks, toks[1:])}
    return {w for w in qw if f(w) in have}


def _trim_reference(items: list[dict], n_words: int) -> list[dict]:
    """Choose which library entries to show. The library always has something that shares a word with the question,
    so showing every hit gave "capital of France" a second answer about a region called Nouvelle-Aquitaine.

    An entry whose TITLE is what the question names wins outright (asking "what is DNA" wants the DNA article, not
    the ones that merely mention it). Otherwise only the best-covering entries are kept. A fact and an article with
    the same title are one answer, and the curated fact is the one kept."""
    def score(c: dict) -> float:
        cov = len(c["overlap"]) / max(1, n_words)
        in_title = len(_covered(set(c["overlap"]), c.get("title") or "")) / max(1, n_words)
        return (cov + 0.4 * in_title + (0.5 if c.get("title_hit") else 0)
                + (0.25 if c.get("topic") == "general knowledge" else 0))
    ranked = sorted(items, key=lambda c: -score(c))
    if not ranked:
        return []
    named = [c for c in ranked if c.get("title_hit")]
    if not named and ranked[0].get("topic") == "general knowledge":
        return ranked[:1]                  # a curated fact is the whole answer; its look-alikes are not
    pool = named or [c for c in ranked if score(c) >= score(ranked[0]) - 0.15]
    out, seen = [], set()
    for c in pool:
        t = (c.get("title") or c["id"]).lower()
        if t in seen:
            continue
        seen.add(t)
        out.append(c)
    return out[:2]


# What counts as a SIMPLE general-knowledge question for the local model. Anything about machines, faults, repairs
# or this device is excluded no matter how it is phrased: a 1.5B model has no business guessing at those.
_GENERAL_START = re.compile(r"^\s*(?:what(?:'s|s)?|who(?:'s|s)?|whom|when|where|which|how (?:many|much|far|old|tall|long|big|large|high|deep|fast)"
                            r"|define|meaning of|capital of|full form of|name |tell me (?:about|a fact|the|what|who|where|when|how)|give me (?:a fact|the)|list |"
                            r"is |are |was |were |do |does |did |can |will |has |have )", re.I)
_MACHINE_STEMS = ("machin", "bearing", "motor", "pump", "fault", "vibrat", "repair", "fix", "sensor", "devic", "episod",
                  "fleet", "engine", "gearbox", "shaft", "fan", "compress", "turbin", "mainten", "technic", "torque",
                  "rpm", "spindle", "coolant", "lubric", "misalign", "unbalanc", "imbalanc", "symptom", "diagnos",
                  "brake", "clutch", "error", "dtc", "obd", "vehicle", "truck", "robot", "calibrat", "taught",
                  "teach", "record", "plant", "site", "signal", "baseline", "novelty", "verif", "cloud", "sync",
                  "mirror", "qdrant", "valve", "gasket", "piston", "cylinder", "hydraul", "pneumat", "conveyor",
                  "weld", "install", "replac", "overheat", "leak", "noise", "wear", "crack", "plc", "scada", "kiosk",
                  "ecu", "batter", "charg", "autonom", "lidar", "radar", "firmware", "protocol", "network", "hmi",
                  "servo", "actuat", "encoder", "kinemat", "cyber", "secur", "safety", "reliab", "phone", "mobile",
                  "circuit", "voltage", "current", "transistor", "inverter", "controller", "automat", "industrial")
_PERSONAL = frozenset("my our your mine we i us this these those here".split())
# Facts that depend on where or when the question is asked (or are private) cannot come from training at all. The
# benchmark showed the model inventing "911" for "phone number of the nearest pharmacy".
_RELATIVE = re.compile(r"\b(nearest|nearby|near me|yesterday|today|tomorrow|tonight|currently|right now|latest|current"
                       r"|this (?:week|month|year|morning|evening)|password|phone number|address|email|price of|weather"
                       r"|temperature in|score)\b", re.I)


def _simple_general(q: str) -> bool:
    """Is this a short factual question about the world, and nothing to do with machines or this device?"""
    words = re.findall(r"[a-z0-9']+", (q or "").lower())
    if not 2 <= len(words) <= 16 or not _GENERAL_START.match((q or "").strip()):
        return False
    # a code like P0301 or plot91 names one thing on this device; a plain number ("boils at 100 degrees") does not
    if (any(re.search(r"[a-z]", t) for t in _identifiers(q)) or _PERSONAL.intersection(words)
            or _RELATIVE.search(q) or _OPEN_ENDED.match(q) or _LABELLED_NUMBER.search(q)):
        return False
    return not any(w.startswith(_MACHINE_STEMS) for w in words)


# Words that appear in nearly every record of a maintenance device ("machine", "causes", "problem") say nothing about
# whether a record is about the question: "what causes rain" was answered with a fleet case about bearings, and "what is
# machine learning" was mixed with episode records, because they share one of these words.
_GENERIC = frozenset("machine machines cause causes caused problem problems issue issues happen happens happened thing "
                     "things good bad new old time example kind type way fix fixed work works working help used use "
                     "device devices site sites record records episode episodes data info information".split())
# question verbs that a correct passage words differently ("wrote" / "written", "invented" / "inventor")
_ASK_VERBS = frozenset("wrote written write writes invented invent discovered discover painted paint founded found built "
                       "build created create called known made make born died located situated named designed".split())
_ENCYCLOPEDIC = ("Simple English Wikipedia", "Technical concepts (Wikipedia)")
# Questions that ask for a design, a calculation, a diagnosis or a judgment have no passage to quote: a lead paragraph
# about the same topic would be a confident non-answer. They are answered from the device's own notes and procedures, or
# not at all ("Needs internet connection for this.").
_OPEN_ENDED = re.compile(
    r"^\s*(?:design|calculate|compute|derive|diagnose|troubleshoot|redesign|explain why"
    r"|how (?:would|should|could|can) (?:you|we|i|one)\b"
    r"|how do (?:you|we) (?:design|diagnose|prove|test|reduce|increase|choose|select|size|calculate|implement|handle"
    r"|detect|secure|validate|verify|decide|optimi[sz]e|build|distinguish|quantify|estimate|measure)\b"
    r"|what (?:if|happens|would|should)\b"
    r"|why (?:can|could|would|should|did|do|does|is|are|was|use|not)\b"
    r"|which\b.*\b(?:better|best|worse|superior|suitable)\b"
    r"|when (?:is|would|should|does) .*\b(?:better|best|worse|superior)\b"
    r"|where should\b)", re.I)
from edge.converse import _COMPARE   # noqa: E402  (one definition of "this asks to compare things")
_DEFINITIONAL = re.compile(r"^\s*(?:what (?:is|are|was|were) (?:an? |the )?|who (?:is|was|are|were) |define |"
                           r"tell me about |explain |how (?:does|do) (?:an? |the )?.+ work)", re.I)
_SOFT_WORDS = frozenset("city name called country place town work works working robot robots motor sensor "
                        "system device protocol network algorithm".split())    # filler a curated fact need not repeat

_LABELLED_NUMBER = re.compile(r"\b(?:plot|unit|line|bay|zone|batch|lot|id|serial|episode|window|step|tag|ticket|record|site|row"
                              r"|device|machine|bearing|pump|motor)\s*#?\d+", re.I)
_NUM_WORDS = (("multiplied by", "*"), ("divided by", "/"), ("to the power of", "**"), ("raised to", "**"),
              ("plus", "+"), ("minus", "-"), ("times", "*"), ("into", "*"), ("over", "/"), ("mod", "%"), ("x", "*"),
              ("×", "*"), ("÷", "/"), ("^", "**"))


def _arithmetic(q: str) -> tuple[str, str] | None:
    """(expression, result) when the whole question is plain arithmetic ("what is 12 times 7", "2+2*3"); else None.
    Evaluated with a restricted parser - numbers and + - * / % ** and brackets only - so it is exact, instant, and cannot
    run anything else. A language model is never asked to do sums."""
    import ast
    import operator as op
    t = (q or "").lower().strip().rstrip("?=. ")
    t = re.sub(r"^(?:what(?:'s|s| is| was)?|calculate|compute|solve|find|evaluate|how much is|tell me)\s+", "", t).strip()
    for word, sym in _NUM_WORDS:
        t = re.sub(rf"(?<=[\d)\s]){re.escape(word)}(?=[\d(\s])" if word.isalpha() else re.escape(word), f" {sym} ", t)
    t = re.sub(r"\s+", " ", t.replace(",", "")).strip()
    if not re.fullmatch(r"[\d\s.+\-*/%()]+", t) or not re.search(r"\d", t) or not re.search(r"[+\-*/%]", t):
        return None
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Mod: op.mod,
           ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos}

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and (abs(b) > 64 or abs(a) > 1e12):
                raise ValueError("too large")
            return ops[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        raise ValueError("not arithmetic")
    try:
        val = ev(ast.parse(t.replace("**", "**"), mode="eval"))
    except Exception:
        return None
    if isinstance(val, float):
        val = round(val, 10)
        val = int(val) if val == int(val) and abs(val) < 1e15 else val
    return t.replace("**", "^"), f"{val:,}" if isinstance(val, int) and abs(val) >= 10000 else str(val)


CHAT_CONTEXT_TURNS = 3
NEEDS_INTERNET = "Needs internet connection for this."

# words that point at something already said instead of naming it. Question words ("how", "why", "what")
# are deliberately NOT here: "how big is plot 91" is a complete question, and treating it as a follow-up
# made it inherit the previous question's subject and answer about plot 44.
_REFERRING = frozenset("""
it its that this these those they them their he she him her there
""".split())
_DEMONSTRATIVE = frozenset("this these those there".split())


# Where a piece of evidence came from, in the four words the UI uses. The local model is never one of them:
# it is not a source, so an answer written from its own training carries no origin at all.
_ORIGINS = {"web": "web", "reference": "offline_kb", "fleet": "fleet"}


def _origin(source: str) -> str:
    return _ORIGINS.get(source, "device")


def _supported(sentence: str, texts: dict[str, str], question: str) -> bool:
    """Is every meaningful word of a model-written sentence found in the evidence it cites (or the question)?

    The grounding check proves a sentence carries a citation, not that the citation says it. This is the cheap,
    conservative proxy for "says it": a sentence that introduces a word the evidence never used is making a
    claim of its own. Words are compared by their first five letters so "painted" matches "painting"."""
    cited = {f"E{n}" for n in re.findall(r"\[E(\d+)\]", sentence)}
    allowed = _content_words(question)
    for k in cited:
        allowed |= _content_words(texts.get(k, ""))
    stems = {w[:5] for w in allowed}
    return all(w[:5] in stems for w in _content_words(re.sub(r"\[E\d+\]", "", sentence)))


def _is_follow_up(q: str) -> bool:
    """Does this question lean on the conversation? Decided once, in edge/converse.py."""
    from edge import converse
    return converse.is_followup(q)


def _attach_lone_citation(raw: str, key: str) -> str:
    """When there is exactly one piece of evidence, mark each sentence with it.

    The model tends to cite once at the end rather than per sentence, and the grounding check works
    sentence by sentence — so a correct answer was being thrown away over where the marker sat. With a
    single source the attribution is not in doubt, and every other check still applies: a sentence with
    a number that is not in the evidence, or that tells the reader what to do, is still dropped. This is
    deliberately not done when several sources are in play, because there "which sentence came from
    which source" is a real question and guessing it would be inventing attribution.
    """
    out = []
    for s in re.split(r"(?<=[.!?])\s+|\n+", raw or ""):
        s = s.strip()
        if not s:
            continue
        bare = re.sub(r"\[E\d+\]", "", s).strip()
        if not bare:                       # a fragment that is only a citation carries no claim
            continue
        if f"[{key}]" in s:
            out.append(s)
            continue
        # the marker goes INSIDE the sentence, before its final punctuation. Appended after the full
        # stop it is split off as the start of the next sentence, which shifts every citation by one
        # and leaves the first sentence looking uncited.
        m = re.match(r"^(.*?)([.!?]+)$", bare, re.S)
        out.append(f"{m.group(1).rstrip()} [{key}]{m.group(2)}" if m else f"{bare} [{key}]")
    return " ".join(out)


def _names_it(text: str, matched: list[str]) -> int:
    """Is this record *about* one of the matched identifiers, rather than just mentioning it?

    Reference entries lead with what they describe ("P0301: Cylinder 1 Misfire Detected."), so an
    identifier in the opening words means the record is that thing.
    """
    head = (text or "")[:60].lower()
    return 1 if any(t in head for t in matched) else 0


def _identifiers(s: str) -> set[str]:
    """Tokens that pick out one specific thing: plot 91, error P0301, bearing 6205, the year 2026.

    These decide relevance on their own. "plot 91" and "plot 44" share every ordinary word, so word
    overlap alone happily answers a question about one with the record for the other — which is worse
    than saying nothing, because it looks like an answer.
    """
    # a lone digit ("atomic number 1", "step 2") names nothing; two digits or a letter-digit mix ("91", "p0301") does
    return {t for t in re.findall(r"[a-z]*\d[a-z0-9]*", (s or "").lower())
            if t not in _STOPWORDS and (len(t) > 1 or not t.isdigit())}


# Questions about the device's record as a whole, rather than about any one thing in it.
# Word overlap cannot answer these: "has anything here been fixed before?" shares no word with an
# episode record, so the four questions the UI itself offers as starting points all came back
# "nothing on this device relates to that at all" while the device held a verified repair. These are
# answered from stored episode state instead of from retrieval. Ordered: the first phrase that matches
# wins, so the narrower intent ("still unresolved") is tested before the broader one ("what problems").
_OVERVIEW_PHRASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("unresolved", ("still unresolved", "unresolved", "still open", "still broken", "not fixed",
                    "never fixed", "outstanding", "still a problem", "still wrong", "anything open",
                    "yet to be fixed")),
    ("fixed",      ("been fixed", "fixed before", "was fixed", "ever fixed", "fixed anything",
                    "what fixed", "what worked", "any fixes", "repaired", "repairs", "sorted out")),
    ("problems",   ("what problems", "any problems", "what faults", "any faults", "what issues",
                    "any issues", "went wrong", "gone wrong", "anything wrong", "what have you seen",
                    "seen recently", "seen lately", "what happened", "anything unusual")),
    ("known",      ("know about me", "know about this", "know about the machine", "what do you know",
                    "what do you remember", "what is in your memory", "whats in your memory",
                    "what have you got", "what do you have", "about yourself")),
)


# Words an overview question is allowed to be made of. A question that names anything beyond these is
# about a specific thing, not about the record as a whole: "what happened to the pump housing" shares the
# phrase "what happened" with "what happened recently", and only the subject it names tells them apart.
_OVERVIEW_VOCAB = frozenset("""
anything everything something nothing problem problems fault faults issue issues trouble wrong
fixed fix fixes repair repairs repaired resolved unresolved outstanding broken sorted worked work
seen saw happened happening recent recently lately unusual odd strange still open pending waiting
know knows remember memory memories hold holds machine machines device devices here there yourself
told taught learned ever never before yet
""".split())


def _overview_intent(q: str) -> str | None:
    """Which whole-record question this is, or None if it asks about something specific.

    Two conditions, both required: the question uses one of the known phrasings, AND it names nothing
    outside the overview vocabulary. The second is what keeps "what happened to the pump housing" on the
    retrieval path, where the stored note about the pump actually answers it.
    """
    s = " " + " ".join(re.sub(r"[^a-z0-9 ]+", " ", (q or "").lower()).split()) + " "
    if _content_words(q) - _OVERVIEW_VOCAB:
        return None
    for intent, phrases in _OVERVIEW_PHRASES:
        if any(p in s for p in phrases):
            return intent
    return None


def _ep_line(e: dict) -> str:
    """One episode, stated as what was measured and recorded - never as a diagnosis.

    The wording keeps the project's rule: the sensor says a symptom resolved, it does not say a root
    cause was confirmed, and nothing here recommends an action.
    """
    fc = e.get("fault_class") or (e.get("fault_hint") or {}).get("fault_class") or "unclassified"
    src = ("confirmed by a technician" if e.get("fault_class_source") == "technician"
           else "suggested by physics, not yet confirmed")
    s = (f"Episode #{e.get('seq')} on {e.get('machine_id')}: {e.get('component')} / {fc} "
         f"({src}), {e.get('occurrences')} window(s), first seen {str(e.get('first_seen', ''))[:16]}.")
    if e.get("action_code"):
        s += f" Action taken: {e['action_code']} (outcome recorded as {e.get('outcome', 'pending')})."
    else:
        s += " No action has been recorded against it yet."
    v = e.get("verify") or {}
    if v.get("verdict") == "symptom_resolved":
        s += (f" The sensor then saw {v.get('consecutive_ok', v.get('required', 0))} consecutive "
              f"healthy windows: symptom resolved.")
    elif v.get("verdict") == "symptom_persists":
        s += " The sensor kept seeing the symptom afterwards: it did not hold."
    elif e.get("status") == "verifying":
        s += (f" Still verifying: {v.get('consecutive_ok', 0)} of {v.get('required', 0)} "
              f"consecutive healthy windows so far.")
    return s


@dataclass
class DeviceConfig:
    device_id: str
    site_id: str
    machine_id: str
    root: pathlib.Path
    machine_class: str = "2hp-induction-motor/SKF6205-DE"
    component: str | None = None                # None = the profile's default component
    denylist: list[str] = field(default_factory=list)
    cloud_url: str | None = None
    device_token: str | None = None
    profile: str = "bearing-12k"                # edge/profiles.py: what kind of signal this device watches
    profile_params: dict = field(default_factory=dict)
    compress_storage: bool = False              # Windows: NTFS-compress the device folder at start + hourly
    encrypt_notes: bool = True                  # technician notes encrypted at rest (edge/crypto.py)
    hold_days: float = 30.0                     # after a verified fix: report 'held' after this, or 'recurred'
    clock_offset_s: float = 0.0                 # SIMULATES a wrong device clock (tests/benchmarks); 0 in production


class Device:
    def __init__(self, cfg: DeviceConfig, embedder: Embedder):
        self.cfg, self.embedder = cfg, embedder
        self.profile = profiles.make(cfg.profile, **cfg.profile_params)
        self.component = cfg.component or self.profile.default_component
        root = pathlib.Path(cfg.root)
        root.mkdir(parents=True, exist_ok=True)
        self.card = machine_card.load(root)          # manufacturer data (bearing, limits, mains), if entered
        self.profile.apply_card(self.card)
        self.store = EdgeStore(root / "local", text_model=embedder.name, note_dim=embedder.dim,
                               fp_version=self.profile.fp_version, allow_text_model_change=True)
        self.note_protection = "none (encrypt_notes off)"
        if cfg.encrypt_notes:
            key, how = load_or_create_key(root)
            self._cipher = NoteCipher(key, how)
            self.store = _NoteVault(self.store, self._cipher)
            self.note_protection = f"AES-256-GCM, key protected by {how}"
        self.outbox = Outbox(str(root / "device.sqlite"))
        self.mirror = Mirror(root, self.outbox, text_model=FLEET_TEXT_MODEL)
        if self.store.pending_text_model:            # H.3: the device got a new text model -> re-embed its memory
            m = self.store.migrate_text_model(embedder.embed_documents,
                                              lambda p: self._doc_text(self._plain(p)) if p.get("type") == "episode" else None)
            self.outbox.log("model", f"text model changed {m['from']} -> {m['to']}: re-embedded {m['migrated']} "
                                     f"episode(s) into a new vector; BM25 and fingerprints unchanged")
        self._lock = threading.RLock()
        self.baseline: Baseline | None = None
        self.gate: NoveltyGate | None = None
        self.counters = collections.Counter()
        self.gate_ms: collections.deque[float] = collections.deque(maxlen=2000)
        self.search_ms: collections.deque[float] = collections.deque(maxlen=500)
        self.last_gate: dict | None = None
        self.recent: collections.deque[dict] = collections.deque(maxlen=240)   # recent window results (UI chart)
        self._run_episode: str | None = None       # episode of the current uninterrupted abnormal run
        self._capture: dict | None = None          # healthy-baseline capture from a live sensor (ingest_signal)
        self.taught: dict[str, list[float]] = {}   # operating variable -> [min, max] the healthy baseline covers
        self.last_diagnosis: dict | None = None    # physics diagnosis of the latest abnormal signal window
        replayed = self._replay_journal()
        requeued = self.outbox.recover_uploading()
        bpath = root / "baseline.json"
        if bpath.exists():
            b = json.loads(bpath.read_text())
            self.taught = b.get("taught", {})
            self.baseline = Baseline.from_dict(b["baseline"], self.profile.fp_version)
            self.gate = NoveltyGate(self.store, cfg.machine_id, GateConfig(**b["gate"]))
        self.outbox.log("boot", f"device started; journal re-applied {replayed} op(s); {requeued} upload(s) re-queued")
        self._last_retention = 0.0
        self.maybe_run_retention()                 # also compresses the folder when compress_storage is on

    # ---- the device's clock ------------------------------------------------------------------------------
    def _now(self) -> dt.datetime:
        """This device's wall clock. Real devices can be minutes or days off; the cloud measures the offset on every
        sync and corrects the times of this device's evidence (cloud/ingest.py), and the device shows the offset."""
        return dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=self.cfg.clock_offset_s)

    def _now_iso(self) -> str:
        return self._now().isoformat(timespec="milliseconds")

    def clock_status(self) -> dict | None:
        off = self.outbox.kv_get("clock_offset_vs_cloud_s")
        if off is None:
            return None
        return {"offset_vs_cloud_s": round(float(off), 1), "ok": abs(float(off)) <= CLOCK_WARN_S,
                "text": ("clock agrees with the fleet cloud" if abs(float(off)) <= CLOCK_WARN_S else
                         f"this device's clock is {abs(off) / 3600:.1f} h {'ahead of' if off > 0 else 'behind'} the "
                         "fleet cloud: shared evidence is time-corrected by the cloud; set the clock (NTP)")}

    # ---- persistence: journal first, then shard ---------------------------------------------------------
    def _apply(self, body: dict) -> None:
        kind = body["kind"]
        if kind == "upsert":
            c = getattr(self, "_cipher", None)
            pts = [StorePoint(**p) for p in body["points"]]
            if c:
                # rebuilt by keyword: a positional rebuild silently drops any field added later, which is
                # exactly what happened to the picture vector the first time
                pts = [StorePoint(p.id, p.payload, p.vib, p.note, c.decrypt(p.bm25_text), p.image)
                       for p in pts]
            self.store.upsert(pts)
        elif kind == "set_payload":
            self.store.modify(body["id"], lambda _p: body["fields"])
        elif kind == "set_payload_where":
            self.store.set_payload_where(body["filter"], body["fields"])
        elif kind == "archive":                  # one journal op, so a crash cannot lose the kept exemplar
            self.store.delete_where({"type": "exemplar", "episode_id": body["episode_id"]})
            if body["keep"]:
                self.store.upsert([StorePoint(**body["keep"])])
            self.store.modify(body["episode_id"], lambda _p: body["fields"])
        else:
            raise ValueError(kind)

    def _write(self, body: dict) -> None:
        op = ids.make_id("op", self.cfg.device_id, time.time_ns(), id(body))
        self.outbox.journal_put(op, body)
        self._apply(body)
        self.outbox.journal_applied(op)

    def _replay_journal(self) -> int:
        pending = self.outbox.journal_pending()
        for op, body in pending:
            self._apply(body)
            self.outbox.journal_applied(op)
        return len(pending)

    def _plain(self, p: dict) -> dict:
        c = getattr(self, "_cipher", None)
        return p | {"note_text": c.decrypt(p["note_text"])} if c and p.get("note_text") else p

    def _seal(self, fields: dict) -> dict:
        """Encrypt note_text BEFORE it reaches the journal (the store wrapper encrypts the shard copy too)."""
        c = getattr(self, "_cipher", None)
        return fields | {"note_text": c.encrypt(fields["note_text"])} if c and fields.get("note_text") else fields

    def _upsert(self, points: list[StorePoint]) -> None:
        c = getattr(self, "_cipher", None)
        # the BM25 text contains the note, so it is journaled encrypted too; _apply decrypts it in memory only
        points = [StorePoint(p.id, self._seal(p.payload), p.vib, p.note,
                             c.encrypt(p.bm25_text) if c and p.bm25_text else p.bm25_text, p.image)
                  for p in points]
        self._write({"kind": "upsert", "points": [p.__dict__ | {"vib": list(p.vib) if p.vib is not None else None,
                                                                 "note": list(p.note) if p.note is not None else None,
                                                                 "image": list(p.image) if p.image is not None else None}
                                                  for p in points]})

    def _set(self, pid: str, **fields) -> None:
        self._write({"kind": "set_payload", "id": pid, "fields": self._seal(fields)})

    # ---- operating points (load / speed) the healthy baseline covers ------------------------------------
    OP_TOLERANCE = 0.02         # relative margin around a taught range (speed jitter, slip)

    def _save_baseline_file(self) -> None:
        (pathlib.Path(self.cfg.root) / "baseline.json").write_text(json.dumps(
            {"baseline": self.baseline.to_dict(), "gate": self.gate.cfg.to_dict(), "taught": self.taught}))

    def _teach_ops(self, ops: list[dict | None]) -> None:
        """Widen the taught range of each operating variable to include these operating points."""
        for op in ops:
            for k, v in (op or {}).items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    lo, hi = self.taught.get(k, [float(v), float(v)])
                    self.taught[k] = [min(lo, float(v)), max(hi, float(v))]

    def _op_suggestion(self, u: dict) -> str:
        """Physics decides the wording: the median fault-signature score of the episode's first windows against the
        profile's threshold (chosen on CWRU; bench/operating_point.py measures it on HUST)."""
        scores = u.get("sig_scores") or []
        if not scores:
            return "this operating point was never taught: check whether the machine is healthy"
        med = float(np.median(scores))
        if med >= self.profile.SIGNATURE_THRESHOLD:
            return (f"a fault signature is present (defect score {med:.2f} >= {self.profile.SIGNATURE_THRESHOLD}) even "
                    "though this operating point was never taught")
        return (f"probably a new normal operating point (no fault signature, score {med:.2f}): if the machine is "
                "healthy, press 'Not a fault: normal operation'")

    def untaught(self, op: dict | None) -> dict:
        """Operating variables whose value lies outside what the healthy baseline was taught (with a margin).
        Empty if nothing is known: no data, no claim."""
        out = {}
        for k, v in (op or {}).items():
            if k in self.taught and isinstance(v, (int, float)):
                lo, hi = self.taught[k]
                m = self.OP_TOLERANCE * max(abs(lo), abs(hi), 1e-9)
                if v < lo - m or v > hi + m:
                    out[k] = {"value": round(float(v), 3), "taught": [round(lo, 3), round(hi, 3)]}
        return out

    # ---- baseline ---------------------------------------------------------------------------------------
    def fit_baseline(self, healthy_raw: np.ndarray, ops: list[dict | None] | None = None) -> dict:
        """Fit z-scoring stats on this machine's healthy windows, store them as baseline points, calibrate the
        gate from them. Replaces any previous baseline of this machine. `ops` (one per window, optional): the
        operating point of each window, e.g. {"speed_hz": 29.9, "load_kw": 1.5}, remembered as taught ranges."""
        with self._lock:
            self.taught = {}
            self._teach_ops(ops or [])
            b = Baseline.fit(healthy_raw, self.profile.fp_version, self.profile.min_std)
            z = b.z(np.asarray(healthy_raw, dtype=np.float64))
            g = calibrate(z, normal_factor=self.profile.normal_factor, target_false_alarm=self.profile.target_false_alarm)
            pts = [StorePoint(ids.baseline_point_id(self.cfg.machine_id, i),
                              {"type": "baseline", "machine_id": self.cfg.machine_id,
                               "fp_version": self.profile.fp_version}, vib=z[i].tolist()) for i in range(len(z))]
            for s in range(0, len(pts), 256):
                self._upsert(pts[s:s + 256])
            self.baseline, self.gate = b, NoveltyGate(self.store, self.cfg.machine_id, g)
            self._save_baseline_file()
            self.outbox.log("baseline", f"baseline fitted on {len(z)} healthy windows; tau_normal={g.tau_normal:.2f}, "
                                        f"tau_merge={g.tau_merge:.2f} (calibration q99 {g.calib_q99:.2f})"
                                        + (f"; taught operating points {self.taught}" if self.taught else ""))
            return g.to_dict() | {"n_windows": len(z), "taught": self.taught}

    # ---- live sensor input (any profile) ----------------------------------------------------------------
    def start_baseline_capture(self, n_windows: int) -> dict:
        """The next n_windows signal windows are treated as THIS machine's healthy state (the technician says the
        machine is known-good now); then the baseline is fitted from them."""
        if n_windows < 10:
            raise ValueError("capture at least 10 windows (the gate calibration needs them)")
        with self._lock:
            self._capture = {"want": int(n_windows), "features": [], "ops": []}
        self.outbox.log("baseline", f"capturing {n_windows} healthy windows from the live sensor")
        return self.capture_state()

    def capture_state(self) -> dict | None:
        c = self._capture
        return None if c is None else {"want": c["want"], "have": len(c["features"])}

    def ingest_signal(self, x, fs: float, rpm: float | None = None, source: str | None = None,
                      operating_point: dict | None = None) -> list[dict]:
        """Raw sensor data (any length; the profile cuts it into windows) -> fingerprint -> gate. While a baseline
        capture runs, windows only feed the capture. Returns one gate result per window. The operating point is
        the shaft speed (from rpm) plus anything the caller knows, e.g. {"load_kw": 1.2}."""
        results = []
        op = {k: float(v) for k, v in (operating_point or {}).items() if isinstance(v, (int, float))}
        if rpm and rpm > 0:
            op["speed_hz"] = rpm / 60.0
        fs_w = self.profile.analysis_fs or fs            # profile.windows() already resampled to the analysis rate
        # order features of the whole chunk at its native rate (fleet-learned hint; needs >= 0.5 s)
        of = None if isinstance(x, dict) else self.profile.order_features(np.asarray(x, dtype=float), fs, rpm)
        touched: set[str] = set()
        for w in self.profile.windows(x, fs) if not isinstance(x, dict) else [x]:
            f = self.profile.features(w, fs_w, rpm)
            self._tm_windows = getattr(self, "_tm_windows", 0) + 1    # readouts seen = windows + 1 (telemetry)
            if not np.all(np.isfinite(f)):
                raise ValueError("signal produced non-finite features (flat or corrupt input?)")
            with self._lock:
                if self._capture is not None:
                    self._capture["features"].append(f)
                    self._capture["ops"].append(op or None)
                    if len(self._capture["features"]) >= self._capture["want"]:
                        feats, ops = np.asarray(self._capture["features"]), self._capture["ops"]
                        self._capture = None
                        results.append({"state": "baseline", "fitted": self.fit_baseline(feats, ops)})
                    else:
                        results.append({"state": "capturing", **self.capture_state()})
                    continue
            r = self.ingest_window(f, source, op or None, velocity=self._velocity(w, fs_w))
            if self.profile.name == "telemetry":             # vehicle early-warning hint (trained on real data)
                self.risk_hint = vehicle_risk.risk(f, self.baseline, float(op.get("age", 0.0)), self._tm_windows + 1)
                if self.risk_hint and r.get("episode_id"):
                    self._set(r["episode_id"], risk_hint=self.risk_hint)
            if r["state"] != "normal" and not isinstance(w, dict):
                self.attach_diagnosis(r.get("episode_id"), w, fs_w, rpm)
            if r["state"] != "normal" and isinstance(w, dict) and r.get("episode_id") and w.get("codes"):
                self._note_codes(r["episode_id"], w.get("codes") or {})
            if r["state"] != "normal" and r.get("episode_id"):
                touched.add(r["episode_id"])
            results.append(r)
        for eid in touched:
            self.attach_order_features(eid, of)
        return results

    def _velocity(self, w, fs: float) -> float | None:
        """Vibration velocity of one window, only while an episode is verifying against a machine-card limit."""
        if self.card is None or isinstance(w, dict) or not self.verifying_with_limit():
            return None
        try:
            v = P.velocity_rms_mm_s(np.asarray(w, dtype=float), fs)
        except (TypeError, ValueError):
            return None
        return float(v) if np.isfinite(v) else None

    def verifying_with_limit(self) -> bool:
        return any((ep.get("verify") or {}).get("limit_mm_s") for ep in self.episodes(status="verifying"))

    def limit_for_verification(self) -> tuple[float | None, str | None]:
        """The acceptable velocity a fix must get below, from the machine card; None when this profile's signal is not
        a calibrated acceleration (phone, microphone, robots, events) or there is no card."""
        if self.card is None or self.profile.name not in ("bearing-12k", "rotating-hf"):
            return None, None
        bounds, ref = self.card.severity_bounds()
        return float(bounds[1]), ref

    # ---- the machine's own learned detector (edge/local_detector.py) --------------------------------------
    def _learned_alarm(self, z) -> float | None:
        if not self.profile.learned_detector:
            return None
        m = self.outbox.kv_get("local_detector")
        if not m:
            return None
        p = local_detector.probability(m, z)
        return p if p > 0.5 else None

    def train_local_detector(self) -> dict:
        """Healthy baseline fingerprints vs exemplars of technician-confirmed fault episodes of this machine."""
        with self._lock:
            normal = [r.vectors["vib"] for r in self.store.scroll(filter={"type": "baseline",
                                                                          "machine_id": self.cfg.machine_id}, with_vectors=True)]
            confirmed = {e["episode_id"] for e in self.episodes() if e.get("fault_class_source") == "technician"
                         and not e.get("dismissed") and e.get("fault_class") not in (None, "unknown")}
            faults = [r.vectors["vib"] for r in self.store.scroll(filter={"type": "exemplar",
                                                                          "machine_id": self.cfg.machine_id}, with_vectors=True)
                      if r.payload.get("episode_id") in confirmed]
            m = local_detector.train(np.asarray(normal, dtype=float), np.asarray(faults, dtype=float))
            self.outbox.kv_set("local_detector", m)
            cv = m["cross_validated"]
            self.outbox.log("detector", f"learned detector trained on this machine: {len(normal)} healthy vs "
                                        f"{len(faults)} confirmed-fault fingerprints ({len(confirmed)} episodes); "
                                        f"cross-validated detection {cv['detection_rate']:.0%}, false alarms "
                                        f"{cv['false_alarm_rate']:.0%}")
            return m

    # ---- fleet-learned fault hint -------------------------------------------------------------------------
    def fleet_model(self) -> dict | None:
        m = self.outbox.kv_get("fleet_hint_model")
        return m if fleet_hint.valid(m) else None

    def attach_order_features(self, eid: str | None, feats: list[float] | None) -> dict | None:
        """Keep the order features of an episode's first chunks (median = robust) and refresh the combined hint."""
        if not eid or feats is None:
            return None
        with self._lock:
            ep = self.store.get(eid)
            if ep is None or ep.payload.get("type") != "episode" or ep.payload["status"] == "closed":
                return None
            seen = list(ep.payload.get("order_feats_seen") or [])
            if len(seen) >= HINT_WINDOWS:
                return ep.payload.get("fleet_hint")
            seen.append([round(float(v), 4) for v in feats])
            med = np.median(np.asarray(seen), axis=0).round(4).tolist()
            h = fleet_hint.combine(ep.payload.get("fault_hint"), med, self.fleet_model())
            self._set(eid, order_feats_seen=seen, order_features=med, fleet_hint=h,
                      bearing=getattr(getattr(self.profile, "geometry", None), "name", None))
            return h

    def refresh_fleet_hints(self) -> int:
        """After a new fleet model arrives: recompute the combined hint of open episodes."""
        n = 0
        with self._lock:
            for ep in self.episodes():
                if ep["status"] != "closed" and ep.get("order_features"):
                    self._set(ep["episode_id"], fleet_hint=fleet_hint.combine(ep.get("fault_hint"), ep["order_features"],
                                                                              self.fleet_model()))
                    n += 1
        return n

    # ---- machine card (manufacturer data) -----------------------------------------------------------------
    def set_machine_card(self, data: dict) -> dict:
        card = machine_card.MachineCard.from_dict(data)
        with self._lock:
            old_geo = getattr(self.profile, "geometry", None)
            prof = profiles.make(self.cfg.profile, **self.cfg.profile_params)
            prof.apply_card(card)
            machine_card.save(self.cfg.root, card)
            self.card, self.profile = card, prof
            new_geo = getattr(prof, "geometry", None)
            recapture = bool(self.baseline is not None and old_geo != new_geo and prof.name != "bearing-12k")
            self.outbox.log("machine_card", f"machine card saved ({card.manufacturer} {card.model}); severity: "
                            f"{card.severity_bounds()[1]}" + ("; bearing geometry changed: RE-CAPTURE the healthy "
                                                               "baseline (the fingerprint uses defect frequencies)"
                                                               if recapture else ""))
            return card.to_dict() | {"baseline_recapture_recommended": recapture}

    def _note_codes(self, eid: str, codes: dict) -> None:
        """Event profile: remember which error codes this episode showed (for the code dictionary and search)."""
        with self._lock:
            ep = self.store.get(eid)
            if ep is None:
                return
            seen = dict(ep.payload.get("codes") or {})
            for c, n in codes.items():
                seen[str(c)[:40]] = seen.get(str(c)[:40], 0) + int(n) if isinstance(n, (int, float)) else 1
            if len(seen) <= 50:
                self._set(eid, codes=seen)

    def attach_diagnosis(self, eid: str | None, raw_window, fs: float, rpm: float | None) -> dict | None:
        """Physics diagnosis of one raw abnormal window, kept on the episode while it is young (first HINT_WINDOWS
        windows), and as `last_diagnosis` for the UI. Used by live input and by the recording replay."""
        d = self.profile.diagnose(raw_window, fs, rpm)
        if not d:
            return None
        self.last_diagnosis = _json_safe(d) | {"episode_id": eid}
        with self._lock:
            ep = self.store.get(eid) if eid else None
            if ep and ep.payload.get("type") == "episode" and ep.payload["occurrences"] <= HINT_WINDOWS:
                self._set(eid, physics=_json_safe(d))
        return d

    # ---- the per-window path ----------------------------------------------------------------------------
    def ingest_window(self, raw: np.ndarray, source: str | None = None, op: dict | None = None,
                      velocity: float | None = None) -> dict:
        if self.gate is None or self.baseline is None:
            raise RuntimeError("fit a healthy baseline first")
        with self._lock:
            self._op = op
            z = self.baseline.z(np.asarray(raw, dtype=float))
            r = self.gate.classify(z)
            learned = self._learned_alarm(z) if r.state == "normal" else None
            if learned is not None:                  # the machine's own learned detector adds an alarm
                r = self.gate.classify_abnormal(z, r)
            self.gate_ms.append(r.latency_ms)
            self.counters["windows"] += 1
            self.counters[r.state] += 1
            ts = self._now_iso()
            out: dict[str, Any] = {"state": r.state, "d_baseline": round(r.d_baseline, 2),
                                   "latency_ms": round(r.latency_ms, 3), "episode_id": r.episode_id, "source": source}
            if learned is not None:
                out["learned_detector"] = round(learned, 3)
            healthy = r.state == "normal"
            new_part = False
            # feed the verifier of every episode that is waiting for its post-action verdict
            for ep in self.episodes(status="verifying"):
                h = healthy
                vs = V.VerifyState.from_dict(ep.get("verify"))
                extra: dict[str, Any] = {}
                if not h and vs.mode == "replacement" and not vs.done:
                    hits = self.store.nearest(z, filter={"type": "exemplar", "episode_id": ep["episode_id"]}, limit=1)
                    if hits and r.d_baseline < hits[0].score:      # nearer to healthy than to the fault
                        h = new_part = True
                        vs.new_part_windows += 1
                        extra["new_part_z"] = [*(ep.get("new_part_z") or [])[-199:], [round(float(v), 5) for v in z]]
                vs = V.step(vs, h, velocity)
                self._set(ep["episode_id"], verify=vs.to_dict(), **extra)
                if vs.done:
                    self.outbox.log("verify", f"{vs.label()}", ep["episode_id"])
                    if vs.verdict == "symptom_resolved" and vs.mode == "replacement":
                        self._adopt_new_part(ep["episode_id"])
                    self._after_verdict(ep["episode_id"])
            if new_part and r.state != "normal":          # a new part's normal: do not open or grow an episode
                self.counters[r.state] -= 1
                self.counters["normal"] += 1
                r.state, r.episode_id = "normal", None
                out.update(state="normal", episode_id=None, new_part=True)
            cont = self._run_episode
            if r.state == "new" and cont and self.store.get(cont).payload["status"] != "closed":
                # Continuity rule: an uninterrupted abnormal run is ONE episode, even if a noisy window lands
                # outside tau_merge of every exemplar (measured: 14-mil ball faults spread up to ~35 z-units).
                r.state, r.episode_id = "merge", cont
                out.update(state="merge", episode_id=cont, merge_reason="contiguous abnormal run")
                self.counters["new"] -= 1
                self.counters["merge"] += 1
            if r.state == "merge":
                self._merge(r.episode_id, z, raw, r.d_episode if r.d_episode is not None else r.d_baseline, ts)
                self._run_episode = r.episode_id
            elif r.state == "new":
                out["episode_id"] = self._open_episode(z, raw, ts, r.recurrence_of)
                out["recurrence_of"] = r.recurrence_of
                self._run_episode = out["episode_id"]
            else:
                self._run_episode = None                         # a healthy window ends the abnormal run
            self.last_gate = out | {"ts": ts}
            self.recent.append({"ts": ts, "state": r.state, "d": round(r.d_baseline, 2)})
            return out

    def _open_episode(self, z: np.ndarray, raw: np.ndarray, ts: str, recurrence_of: str | None) -> str:
        seq = int(self.outbox.kv_get("episode_seq", 0)) + 1
        self.outbox.kv_set("episode_seq", seq)
        eid = ids.episode_id(self.cfg.device_id, seq)
        hint = self.profile.hint(raw, z)
        payload = {
            "type": "episode", "episode_id": eid, "seq": seq, "machine_id": self.cfg.machine_id,
            "device_id": self.cfg.device_id, "site_id": self.cfg.site_id, "component": self.component,
            "profile": self.profile.name,
            "first_seen": ts, "last_seen": ts, "occurrences": 1, "status": "open", "n_exemplars": 1,
            "fault_hint": hint, "hint_votes": {hint["fault_class"]: 1}, "fault_class": None, "fault_class_source": None,
            "action_code": None, "root_cause_claim": None, "action_at": None, "note_text": "", "note_share_opt_in": False,
            "outcome": "pending", "technician_confirmed": False, "verify": None, "share_state": "local",
            "decision": None, "recurrence_of": recurrence_of, "schema_version": 1, "fp_version": self.profile.fp_version,
            "text_model": self.embedder.name, "version": 1,
        }
        op = getattr(self, "_op", None)
        if op:
            payload["operating_point"] = {k: round(v, 4) for k, v in op.items()}
            new_op = self.untaught(op)
            if new_op:
                s = self.profile.signature_score(raw)
                payload["untaught_operating_point"] = new_op | {"sig_scores": [] if s is None else [round(s, 3)]}
                payload["untaught_operating_point"]["suggestion"] = self._op_suggestion(payload["untaught_operating_point"])
        text = self._doc_text(payload)
        self._upsert([
            StorePoint(eid, payload, vib=z.tolist(), note=self.embedder.embed_documents([text])[0], bm25_text=text),
            StorePoint(ids.make_id("exemplar", eid, 0), {"type": "exemplar", "machine_id": self.cfg.machine_id,
                                                         "episode_id": eid, "episode_active": True}, vib=z.tolist()),
        ])
        msg = "unfamiliar state: new episode opened"
        if recurrence_of:
            msg += f" (resembles closed episode {recurrence_of[:8]} on this machine)"
        if payload.get("untaught_operating_point"):
            msg += "; UNTAUGHT operating point: " + payload["untaught_operating_point"]["suggestion"]
        self.outbox.log("gate", msg + f"; physics hint: {hint['fault_class']}", eid)
        self._decide(eid)
        self.check_followups()
        return eid

    def _merge(self, eid: str, z: np.ndarray, raw: np.ndarray, d_episode: float | None, ts: str) -> None:
        ep = self.store.get(eid).payload
        fields: dict[str, Any] = {"occurrences": ep["occurrences"] + 1, "last_seen": ts}
        u = ep.get("untaught_operating_point")
        if u and ep["occurrences"] < HINT_WINDOWS:              # refine the operating-point suggestion (median)
            s = self.profile.signature_score(raw)
            if s is not None:
                u = u | {"sig_scores": [*u.get("sig_scores", []), round(s, 3)]}
                fields["untaught_operating_point"] = u | {"suggestion": self._op_suggestion(u)}
        if ep["occurrences"] < HINT_WINDOWS:                   # majority vote of the first windows' hints
            votes = dict(ep.get("hint_votes") or {})
            h = self.profile.hint(raw, z)
            votes[h["fault_class"]] = votes.get(h["fault_class"], 0) + 1
            best = max(votes, key=votes.get)
            fields["hint_votes"] = votes
            fields["fault_hint"] = ep["fault_hint"] | {
                "fault_class": best, "measured_accuracy": self.profile.measured(best),
                "why": f"{h['why'] if best == h['fault_class'] else 'physics rule'}; "
                       f"voted in {votes[best]} of the first {sum(votes.values())} windows"}
        if d_episode is not None and d_episode > self.gate.cfg.tau_normal and ep["n_exemplars"] < MAX_EXEMPLARS:
            self._upsert([StorePoint(ids.make_id("exemplar", eid, ep["n_exemplars"]),
                                     {"type": "exemplar", "machine_id": self.cfg.machine_id, "episode_id": eid,
                                      "episode_active": True}, vib=z.tolist())])
            fields["n_exemplars"] = ep["n_exemplars"] + 1
        self._set(eid, **fields)

    # ---- technician actions -----------------------------------------------------------------------------
    def _doc_text(self, ep: dict) -> str:
        fc = ep.get("fault_class") or (ep.get("fault_hint") or {}).get("fault_class", "unknown")
        parts = [ep.get("component", ""), fc.replace("_", " "), "fault"]
        if ep.get("action_code"):
            parts += ["action", ep["action_code"].replace("_", " ")]
        if ep.get("outcome") and ep["outcome"] != "pending":
            parts += ["outcome", ep["outcome"]]
        if ep.get("note_text"):
            parts.append(ep["note_text"])
        return " ".join(parts)

    def _rewrite(self, eid: str, **fields) -> dict:
        """Full re-write of an episode point (payload + refreshed text vectors). Returns the new payload."""
        rec = self.store.get(eid, with_vectors=True)
        if rec is None:
            raise KeyError(eid)
        p = rec.payload | fields | {"version": rec.payload.get("version", 1) + 1}
        text = self._doc_text(p)
        self._upsert([StorePoint(eid, p, vib=rec.vectors["vib"], note=self.embedder.embed_documents([text])[0],
                                 bm25_text=text)])
        return p

    def _require(self, eid: str, expected_version: int | None = None) -> dict:
        """The episode payload. With expected_version (the version the technician's screen showed), refuse the
        edit if the episode has moved on: an optimistic compare-and-set, checked under the device lock."""
        rec = self.store.get(eid)
        if rec is None or rec.payload.get("type") != "episode":
            raise KeyError(f"no episode {eid}")
        cur = int(rec.payload.get("version", 1))
        if expected_version is not None and int(expected_version) != cur:
            self.outbox.log("conflict", f"edit refused: expected version {expected_version}, current {cur}", eid)
            raise VersionConflict(eid, int(expected_version), cur)
        return rec.payload

    def set_note(self, eid: str, text: str, share_opt_in: bool = False, expected_version: int | None = None) -> dict:
        with self._lock:
            self._require(eid, expected_version)
            if len(text) > 2000:
                raise ValueError("note too long (max 2000 characters)")
            self._rewrite(eid, note_text=text.strip(), note_share_opt_in=bool(share_opt_in))
            self.outbox.log("note", f"note saved ({len(text)} chars, share opt-in={bool(share_opt_in)})", eid)
            return self._decide(eid)

    def set_fault_class(self, eid: str, fault_class: str, expected_version: int | None = None,
                        damage_mode: str | None = None) -> dict:
        """The technician confirms the fault class, ideally from what they SAW (ISO 15243 damage mode on the removed
        bearing). This confirmed class - never the hint - is what the fleet groups by and learns from."""
        with self._lock:
            self._require(eid, expected_version)
            FaultClass(fault_class)
            if damage_mode:
                DamageMode(damage_mode)
            self._rewrite(eid, fault_class=fault_class, fault_class_source="technician", damage_mode=damage_mode or None)
            self.outbox.log("fault_class", f"technician confirmed fault class {fault_class}"
                            + (f" (seen: {damage_mode.replace('_', ' ')}, ISO 15243)" if damage_mode else ""), eid)
            return self._decide(eid)

    def record_action(self, eid: str, action_code: str, root_cause: str | None = None,
                      required_windows: int = V.REQUIRED_WINDOWS, expected_version: int | None = None) -> dict:
        """An intervention was made: from now on, windows are fed to this episode's outcome verifier."""
        with self._lock:
            ep = self._require(eid, expected_version)
            if ep["status"] in TERMINAL:
                raise ValueError("episode is closed")
            ActionCode(action_code)
            if root_cause:
                RootCause(root_cause)
            lim, src = self.limit_for_verification()
            mode = "replacement" if action_code in V.REPLACEMENT_ACTIONS else "same_part"
            self._rewrite(eid, action_code=action_code, root_cause_claim=root_cause or None, action_at=self._now_iso(),
                          status="verifying", verify=V.VerifyState(required=int(required_windows), limit_mm_s=lim,
                                                                   limit_source=src, mode=mode).to_dict(),
                          outcome="pending", technician_confirmed=False)
            self.store.set_payload_where({"type": "exemplar", "episode_id": eid}, {"episode_active": True})
            self.outbox.log("action", f"action {action_code} recorded; verifying over {required_windows} windows", eid)
            return self._decide(eid)

    def confirm_outcome(self, eid: str, outcome: str, expected_version: int | None = None) -> dict:
        with self._lock:
            ep = self._require(eid, expected_version)
            if outcome not in ("worked", "failed"):
                raise ValueError("outcome must be worked or failed")
            if not ep.get("action_code"):
                raise ValueError("record an action before confirming an outcome")
            self._rewrite(eid, outcome=outcome, technician_confirmed=True)
            self.outbox.log("outcome", f"technician reports the action {outcome}", eid)
            self._after_verdict(eid)
            return self._decide(eid)

    def _after_verdict(self, eid: str) -> None:
        ep = self.store.get(eid).payload
        vs = V.VerifyState.from_dict(ep.get("verify"))
        if vs.done and ep.get("technician_confirmed") and ep["status"] != "closed":
            self._set(eid, status="closed", closed_at=self._now_iso())
            self.store.set_payload_where({"type": "exemplar", "episode_id": eid}, {"episode_active": False})
            self.outbox.log("episode", f"episode closed ({vs.label()})", eid)
        self._decide(eid)

    def _adopt_new_part(self, eid: str) -> int:
        """A replacement verified by the nearest-state rule: its windows become extra healthy-baseline points, so the
        new part's normal is normal from now on (z-normalisation not refitted, like mark_normal)."""
        ep = self.store.get(eid).payload
        zs = ep.get("new_part_z") or []
        pts = [StorePoint(ids.make_id("baseline-newpart", self.cfg.machine_id, eid, k),
                          {"type": "baseline", "machine_id": self.cfg.machine_id, "fp_version": self.profile.fp_version,
                           "regime_from_episode": eid, "new_part": True}, vib=list(v)) for k, v in enumerate(zs)]
        if pts:
            self._upsert(pts)
            self._set(eid, new_part_z=None, new_part_adopted=len(pts))
            self.outbox.log("baseline", f"new part verified: {len(pts)} of its windows added to this machine's healthy "
                                        "baseline", eid)
        return len(pts)

    # ---- policy -----------------------------------------------------------------------------------------
    def _decide(self, eid: str) -> dict:
        rec = self.store.get(eid, with_vectors=True)
        ep = rec.payload
        if ep.get("dismissed"):                           # not a fault: nothing to share, ever
            d = {"action": "KEEP_LOCAL", "event_id": None, "note_shared": False,
                 "reasons": [{"gate": "human", "ok": True, "detail": "marked as normal operation by the technician; "
                              "its fingerprints now extend the healthy baseline"}]}
            self._set(eid, decision=d)
            return d
        red = redact(ep["note_text"], self.cfg.denylist) if ep.get("note_text") else None
        d = policy.decide(ep, fingerprint=rec.vectors["vib"], redaction=red, device_id=self.cfg.device_id,
                          machine_class=self.cfg.machine_class,
                          already_queued=self.outbox.known_event_ids() - {ep.get("event_id")},   # own event is not a duplicate
                          already_repairs={h: e for h, e in self.outbox.known_repairs().items() if e != eid},
                          fp_version=self.profile.fp_version)
        fields: dict[str, Any] = {"decision": d.to_dict()}
        if d.action == "SHARE" and self.outbox.enqueue(d.event):
            fields["share_state"] = "queued"
            fields["event_id"] = d.event["event_id"]
            self.outbox.log("policy", "SHARE: outcome evidence queued for the fleet" +
                            (" (with redacted note)" if d.note_shared else " (note kept local)"), eid)
        elif ep.get("decision") is None or ep["decision"].get("action") != d.action:
            last = next((r for r in reversed(d.reasons) if not r.ok), d.reasons[-1])
            self.outbox.log("policy", f"{d.action}: {last.detail}", eid)
        self._set(eid, **fields)
        return fields["decision"]

    # ---- "this WAS a fault": teach a failure the gate did not flag ----------------------------------------
    def teach_fault(self, raw: np.ndarray, fault_class: str, note: str = "") -> dict:
        """The technician saw a failure (e.g. the robot collided) in a cycle the gate called normal. The window is
        stored as a confirmed-fault example (a closed, 'taught' episode: never shared, no action), so the machine's
        learned detector (edge/local_detector.py) can learn it and the gate recognises it as a recurrence later.
        The mirror image of mark_normal()."""
        if self.gate is None or self.baseline is None:
            raise RuntimeError("fit a healthy baseline first")
        FaultClass(fault_class)
        with self._lock:
            z = self.baseline.z(np.asarray(raw, dtype=float))
            seq = int(self.outbox.kv_get("episode_seq", 0)) + 1
            self.outbox.kv_set("episode_seq", seq)
            eid = ids.episode_id(self.cfg.device_id, seq)
            ts = self._now_iso()
            payload = {"type": "episode", "episode_id": eid, "seq": seq, "machine_id": self.cfg.machine_id,
                       "device_id": self.cfg.device_id, "site_id": self.cfg.site_id, "component": self.component,
                       "profile": self.profile.name, "first_seen": ts, "last_seen": ts, "occurrences": 1,
                       "status": "closed", "closed_at": ts, "n_exemplars": 1,
                       "fault_hint": {"fault_class": "unknown", "why": "taught by the technician"}, "hint_votes": {},
                       "fault_class": fault_class, "fault_class_source": "technician", "action_code": None,
                       "root_cause_claim": None, "action_at": None, "note_text": note.strip()[:2000],
                       "note_share_opt_in": False, "outcome": "pending", "technician_confirmed": True, "verify": None,
                       "share_state": "local", "decision": None, "recurrence_of": None, "schema_version": 1,
                       "fp_version": self.profile.fp_version, "text_model": self.embedder.name, "version": 1,
                       "taught_fault": True}
            text = self._doc_text(payload)
            self._upsert([
                StorePoint(eid, payload, vib=z.tolist(), note=self.embedder.embed_documents([text])[0], bm25_text=text),
                StorePoint(ids.make_id("exemplar", eid, 0), {"type": "exemplar", "machine_id": self.cfg.machine_id,
                                                             "episode_id": eid, "episode_active": False}, vib=z.tolist())])
            self.outbox.log("teach", f"technician taught a {fault_class.replace('_', ' ')} the gate had called normal "
                                     "(kept on this device as a learning example)", eid)
            return self._decide(eid)

    # ---- "not a fault": teach a new healthy operating state ----------------------------------------------
    def mark_normal(self, eid: str, expected_version: int | None = None) -> dict:
        """The technician says this episode is normal operation (a new load, speed or a re-mounted sensor), not a
        fault. Its stored fingerprints become extra healthy-baseline points of this machine (the z-normalisation is
        NOT refitted, so every stored vector stays comparable); the episode closes as dismissed and is never shared.
        Found necessary by the HUST held-out test: healthy data at an untaught load looked abnormal."""
        with self._lock:
            ep = self._require(eid, expected_version)
            if ep["status"] == "closed":
                raise ValueError("episode is closed")
            ex = [r for r in self.store.scroll(filter={"type": "exemplar", "episode_id": eid}, with_vectors=True)]
            pts = [StorePoint(ids.make_id("baseline-extra", self.cfg.machine_id, eid, k),
                              {"type": "baseline", "machine_id": self.cfg.machine_id, "fp_version": self.profile.fp_version,
                               "regime_from_episode": eid}, vib=r.vectors["vib"]) for k, r in enumerate(ex)]
            if pts:
                self._upsert(pts)
            self._write({"kind": "archive", "episode_id": eid, "keep": None,
                         "fields": {"status": "closed", "closed_at": self._now_iso(), "n_exemplars": 0,
                                    "dismissed": {"reason": "normal operation (not a fault)", "at": self._now_iso(),
                                                  "baseline_points_added": len(pts)}}})
            self._teach_ops([ep.get("operating_point"), getattr(self, "_op", None) if self._run_episode == eid else None])
            if self.gate is not None:
                self._save_baseline_file()
            self.outbox.log("baseline", f"technician: normal operation, not a fault; {len(pts)} fingerprint(s) added "
                                        f"to this machine's healthy baseline"
                                        + (f"; taught operating points now {self.taught}" if self.taught else ""), eid)
            self._run_episode = None
            return self._decide(eid)

    # ---- retention (docs/RESEARCH.md G.1 / G.8: separate from the share decision) -----------------------
    def run_retention(self, now: dt.datetime | None = None) -> dict:
        """ARCHIVE closed, decided episodes older than policy.ARCHIVE_AFTER_DAYS: keep the episode point and its
        first exemplar (so a recurrence is still recognised), drop the other exemplar fingerprints. Knowledge is
        never deleted; open, undecided or not-yet-uploaded episodes are never touched."""
        now = now or self._now()
        archived, checked = [], 0
        with self._lock:
            for ep in self.episodes(status="closed"):
                checked += 1
                if ep.get("archived") or ep.get("share_state") in ("queued", "uploading"):
                    continue
                if policy.retention(ep, now) != "ARCHIVE":
                    continue
                eid = ep["episode_id"]
                keep = self.store.get(ids.make_id("exemplar", eid, 0), with_vectors=True)
                self._write({"kind": "archive", "episode_id": eid,
                             "keep": {"id": keep.id, "payload": keep.payload, "vib": keep.vectors["vib"]} if keep else None,
                             "fields": {"archived": True, "archived_at": now.isoformat(timespec="seconds"),
                                        "n_exemplars": 1 if keep else 0}})
                self.outbox.log("retention", f"ARCHIVE: closed + decided + last seen over {policy.ARCHIVE_AFTER_DAYS} "
                                             f"days ago; kept the episode and 1 exemplar", eid)
                archived.append(eid)
        self._last_retention = time.time()
        return {"checked": checked, "archived": archived}

    # ---- did the fix HOLD? (follow-ups weeks after a verified fix) ------------------------------------------
    def check_followups(self, now: dt.datetime | None = None) -> list[dict]:
        """For every shared 'worked' fix without a follow-up: 'recurred' if this machine opened a new episode that
        resembles it (the gate's recurrence match) or carries the same confirmed fault class within hold_days of the
        action; 'held' once hold_days passed without that. One follow-up per fix, queued in the outbox."""
        now = now or self._now()
        out = []
        with self._lock:
            eps = self.episodes()
            for ep in eps:
                if (ep.get("outcome") != "worked" or not ep.get("event_id") or ep.get("followup")
                        or not ep.get("action_at") or self.outbox.status_of(ep["event_id"]) in (None, "rejected")):
                    continue
                t0 = dt.datetime.fromisoformat(ep["action_at"])
                later = [e for e in eps if e["seq"] > ep["seq"] and not e.get("dismissed") and (
                    e.get("recurrence_of") == ep["episode_id"]
                    or (ep.get("fault_class") and e.get("fault_class") == ep["fault_class"]))]
                recur = next((e for e in sorted(later, key=lambda e: e["first_seen"])
                              if dt.datetime.fromisoformat(e["first_seen"]) >= t0), None)
                days_recur = ((dt.datetime.fromisoformat(recur["first_seen"]) - t0).total_seconds() / 86400
                              if recur else None)
                if recur is not None and days_recur <= self.cfg.hold_days:
                    status, days = "recurred", days_recur
                elif (now - t0).total_seconds() / 86400 >= self.cfg.hold_days:
                    status, days = "held", self.cfg.hold_days
                else:
                    continue
                fu = FollowUp(event_id=ids.followup_id(self.cfg.device_id, ep["event_id"]), episode_id=ep["episode_id"],
                              refers_to=ep["event_id"], status=status, days_after_fix=round(max(days, 0.0), 3),
                              hold_days=self.cfg.hold_days, occurred_at=self._now_iso()).model_dump(mode="json")
                self.outbox.enqueue(fu)
                self._set(ep["episode_id"], followup={
                    "status": status, "days_after_fix": fu["days_after_fix"], "hold_days": self.cfg.hold_days,
                    "event_id": fu["event_id"],
                    "recurrence_episode": recur["episode_id"] if recur is not None and status == "recurred" else None})
                self.outbox.log("followup", (f"fix HELD for {self.cfg.hold_days:g} days" if status == "held" else
                                             f"fault RECURRED {fu['days_after_fix']:.1f} days after the fix")
                                + "; follow-up queued for the fleet", ep["episode_id"])
                out.append(fu)
        return out

    def maybe_run_retention(self) -> dict | None:
        """Hourly housekeeping: retention, follow-ups, and (if enabled) re-compressing files Edge created since last
        time."""
        if time.time() - self._last_retention >= RETENTION_EVERY_S:
            self.check_followups()
            out = self.run_retention()
            if self.cfg.compress_storage:
                out["compression"] = storage_os.enable_compression(self.cfg.root)
            return out
        return None

    # ---- usefulness feedback (G.1 "Evaluate usefulness": a counter, no learned model) ---------------------
    def record_feedback(self, result_id: str, kind: str, helped: bool) -> dict:
        """The technician marks a search result as helpful or not. Stored on this device only (SQLite kv) and
        shown next to the result; it never changes ranking and is not shared."""
        key = self._feedback_key(result_id, kind)
        with self._lock:
            fb = self.outbox.kv_get(key, {"helped": 0, "not_helped": 0})
            fb["helped" if helped else "not_helped"] += 1
            self.outbox.kv_set(key, fb)
        self.outbox.log("feedback", f"{kind} result {result_id[:8]} marked {'helpful' if helped else 'not helpful'}")
        return fb

    @staticmethod
    def _feedback_key(result_id: str, kind: str) -> str:
        if kind not in ("local", "fleet"):
            raise ValueError("kind must be local or fleet")
        return f"feedback:{kind}:{canonical_id(result_id)}"          # ValueError if not a point id

    def feedback_for(self, result_id: str, kind: str) -> dict:
        return self.outbox.kv_get(self._feedback_key(result_id, kind), {"helped": 0, "not_helped": 0})

    # ---- reads ------------------------------------------------------------------------------------------
    def episodes(self, status: str | None = None) -> list[dict]:
        flt: dict[str, Any] = {"type": "episode", "machine_id": self.cfg.machine_id}
        if status:
            flt["status"] = status
        eps = [r.payload for r in self.store.scroll(filter=flt)]
        return sorted(eps, key=lambda e: e["seq"], reverse=True)

    def episode(self, eid: str) -> dict:
        ep = self._require(eid)
        return ep | {"outbox_status": self.outbox.status_of(ep["event_id"]) if ep.get("event_id") else None}

    # ---- retrieval (offline: local shard + fleet mirror shard) ------------------------------------------
    # ---- free-text memory: anything the user tells this device, plus its own conversation ----------------
    MEMORY_MAX_CHARS = 4000

    def remember(self, text: str, kind: str = "fact", meta: dict | None = None) -> dict:
        """Store an arbitrary piece of text as searchable memory on this device.

        kind='fact'  something the user taught it, in any domain
        kind='chat'  one turn of the conversation, so follow-up questions have context

        The text goes in `note_text`, which means it inherits the same encryption at rest and the same
        never-leaves-the-device rule as a technician note. It is never synced.
        """
        text = (text or "").strip()
        if not 1 <= len(text) <= self.MEMORY_MAX_CHARS:
            raise ValueError(f"memory text must be 1..{self.MEMORY_MAX_CHARS} characters")
        if kind not in ("fact", "chat", "learned"):
            raise ValueError("kind must be 'fact', 'chat' or 'learned'")
        ts = self._now_iso()
        # 'learned' is keyed on its source URL rather than on the timestamp: looking the same thing up
        # twice must update the one copy, not fill the device with duplicates of it.
        mid = (ids.make_id("memory", self.cfg.device_id, kind, (meta or {}).get("url") or text)
               if kind == "learned" else ids.make_id("memory", self.cfg.device_id, kind, text, ts))
        payload = {"type": "memory", "kind": kind, "note_text": text, "created_at": ts,
                   "device_id": self.cfg.device_id, "machine_id": self.cfg.machine_id,
                   "share_state": "local"} | (meta or {})
        if kind == "chat":
            payload["session"] = (meta or {}).get("session") or self.current_session()
        with self._lock:
            self._upsert([StorePoint(mid, payload,
                                     note=self.embedder.embed_documents([text])[0], bm25_text=text)])
        if kind == "fact":
            self.outbox.log("memory", f"remembered: {text[:70]}")
        return {"id": mid, "kind": kind, "text": text, "created_at": ts}

    def recall(self, text: str, limit: int = 5, kind: str | None = None) -> list[dict]:
        """Semantic search over this device's free-text memory only (facts + past conversation)."""
        flt: dict = {"type": "memory", "device_id": self.cfg.device_id}
        if kind:
            flt["kind"] = kind
        hits = self.store.search(note=self.embedder.embed_query(text), text=text, limit=limit, filter=flt)
        return [{"id": h.id, "score": round(h.score, 4)} | self._plain(h.payload) for h in hits]

    def store_shared_knowledge(self, items: list[dict]) -> int:
        """Store records pulled from the cloud so they stay searchable with the network off.

        These are `type='shared'`, kept apart from `type='memory'` (what this device's own user typed) so the
        answer can say where a fact came from. Withdrawn records are deleted rather than kept and hidden: a
        device that goes offline for a month should not still be answering from something retracted, and the
        copy it already has is the only copy it can act on.
        """
        keep, drop = [], []
        for it in items or []:
            rid, text = it.get("id"), (it.get("text") or "").strip()
            if not rid:
                continue
            # status first: a withdrawn record arrives with its text blanked, so an empty-text check
            # placed above this one would swallow the tombstone and the copy would never be deleted
            if it.get("status") != "active":
                drop.append(rid)
                continue
            if not text:
                continue
            keep.append(StorePoint(rid, {
                "type": "shared", "record_id": rid, "note_text": text, "topic": it.get("topic", "general"),
                "audience": it.get("audience", "everyone"), "author_device": it.get("author_device"),
                "created_at": it.get("updated_at"), "seq": it.get("seq"),
                "device_id": self.cfg.device_id, "share_state": "received",
            }, note=self.embedder.embed_documents([text])[0], bm25_text=text))
        with self._lock:
            if keep:
                self._upsert(keep)
            for rid in drop:
                # delete_where filters on the payload, so the record id has to be a payload field too
                self.store.delete_where({"type": "shared", "record_id": rid})
        if keep:
            self.outbox.log("knowledge", f"received {len(keep)} shared record(s) from the fleet")
        return len(keep)

    def shared_knowledge(self, limit: int = 100) -> list[dict]:
        rows = [self._plain(r.payload) | {"id": r.id}
                for r in self.store.scroll(filter={"type": "shared", "device_id": self.cfg.device_id})]
        rows.sort(key=lambda r: r.get("seq") or 0, reverse=True)
        return rows[:limit]

    def list_memories(self, kind: str | None = None, limit: int = 50) -> list[dict]:
        """Every free-text memory, newest first. Listing is a scroll, not a search: an empty query has
        nothing to rank by, and a blank-string search would return an arbitrary order."""
        flt: dict = {"type": "memory", "device_id": self.cfg.device_id}
        if kind:
            flt["kind"] = kind
        rows = [self._plain(r.payload) | {"id": r.id} for r in self.store.scroll(filter=flt)]
        rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return rows[:limit]

    # ---- pictures as memory (shared/image_embed.py) -----------------------------------------------------
    @property
    def image_embedder(self):
        """Built on first use so a device that never stores a picture never loads the CLIP models."""
        if getattr(self, "_img_embedder", None) is None:
            from shared.image_embed import ImageEmbedder
            self._img_embedder = ImageEmbedder()
        return self._img_embedder

    def remember_image(self, path: str, note: str = "", episode_id: str | None = None) -> dict:
        """Store one picture so it can be found again later.

        The picture itself never leaves the device. What is stored is its CLIP vector plus whatever the
        person typed alongside it — and the typed note is what carries meaning, because nothing here
        claims to know what the picture shows.
        """
        p = pathlib.Path(path)
        if not p.exists():
            raise ValueError(f"no such image: {path}")
        vec = self.image_embedder.embed_images([str(p)])
        if not vec:
            raise RuntimeError("could not read that image")
        ts = self._now_iso()
        pid = ids.make_id("picture", self.cfg.device_id, p.name, ts)
        text = (note or "").strip()
        payload = {"type": "picture", "note_text": text, "file": str(p), "filename": p.name,
                   "created_at": ts, "device_id": self.cfg.device_id, "machine_id": self.cfg.machine_id,
                   "episode_id": episode_id, "share_state": "local"}
        with self._lock:
            self._upsert([StorePoint(pid, payload,
                                     note=self.embedder.embed_documents([text])[0] if text else None,
                                     bm25_text=text or None, image=vec[0])])
        self.outbox.log("picture", f"stored a picture{': ' + text[:50] if text else ''}")
        return {"id": pid, "file": str(p), "note": text, "created_at": ts}

    def recall_images(self, text: str | None = None, like_image: str | None = None,
                      limit: int = 5) -> list[dict]:
        """Find stored pictures, by typed words or by another picture.

        The CLIP text half places words in the picture space, so "cracked housing" can match a photograph
        nobody ever labelled with those words. The result is a list of pictures this device already holds;
        it is not a statement about what any of them shows.
        """
        img_vec = None
        if like_image:
            got = self.image_embedder.embed_images([like_image])
            img_vec = got[0] if got else None
        elif text:
            img_vec = self.image_embedder.embed_query(text)
        if img_vec is None and not text:
            raise ValueError("give text or an image to search with")
        hits = self.store.search(image=img_vec, text=text if img_vec is None else None, limit=limit,
                                 filter={"type": "picture", "device_id": self.cfg.device_id})
        return [{"id": h.id, "score": round(h.score, 4)} | self._plain(h.payload) for h in hits]

    def pictures(self, limit: int = 100) -> list[dict]:
        rows = [self._plain(r.payload) | {"id": r.id}
                for r in self.store.scroll(filter={"type": "picture", "device_id": self.cfg.device_id})]
        rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return rows[:limit]

    def reference_by_id(self, ref_ids: list[str]) -> list[dict]:
        """Exact lookup by reference id, e.g. `dtc:P0420`."""
        # Point ids are deterministic (edge/seed.py: make_id("reference", device, ref_id)), so this is a direct read.
        # Filtering on ref_id instead scanned all 30,000 library records per lookup - about a second per question.
        want = [ids.make_id("reference", self.cfg.device_id, rid) for rid in ref_ids[:6]]
        return [self._plain(r.payload) | {"id": r.id, "score": 1.0} for r in self.store.retrieve(want)
                if r.payload.get("type") == "reference"]

    def recall_reference(self, text: str, limit: int = 5) -> list[dict]:
        """Search the reference packs loaded at first boot (edge/seed.py).

        A code named in the question is looked up exactly first. Among ~9.5k near-identical entries the
        dense leg is close to noise and fusion buries the one exact match, so asking about P0420 came
        back with P0422 — which mentions P0420 in its description. Semantic search still runs, for the
        questions that describe a symptom instead of naming a code.
        """
        exact = self.reference_by_id([f"dtc:{t.upper()}" for t in _identifiers(text)])
        # An entry whose title the question names ("what is dna", "capital of tamilnadu") is looked up by that
        # title. Similarity search alone buries it: any article that merely mentions DNA scores about as well.
        from edge import seed
        try:
            named = self.reference_by_id([rid for rid, _ in seed.title_ref_ids(text)])
        except Exception:                          # the shortcut failing must never cost the ordinary search
            named = []
        for n in named:
            n["title_hit"] = True
        seen = {e["id"] for e in exact} | {n["id"] for n in named}
        hits = self.store.search(note=self.embedder.embed_query(text), text=text, limit=limit,
                                 filter={"type": "reference", "device_id": self.cfg.device_id})
        fuzzy = [{"id": h.id, "score": round(h.score, 4)} | self._plain(h.payload)
                 for h in hits if h.id not in seen]
        return (exact + named + fuzzy)[:max(limit, len(exact) + len(named))]

    def recall_shared(self, text: str, limit: int = 5) -> list[dict]:
        """Search knowledge published by other devices that this one was allowed to receive."""
        hits = self.store.search(note=self.embedder.embed_query(text), text=text, limit=limit,
                                 filter={"type": "shared", "device_id": self.cfg.device_id})
        return [{"id": h.id, "score": round(h.score, 4)} | self._plain(h.payload) for h in hits]

    def recent_chat(self, limit: int = 8, session: str | None = None) -> list[dict]:
        """The last few conversation turns, newest last — used to resolve follow-up questions."""
        rows = [self._plain(r.payload) | {"id": r.id}
                for r in self.store.scroll(filter={"type": "memory", "kind": "chat",
                                                   "session": session or self.current_session(),
                                                   "device_id": self.cfg.device_id})]
        rows.sort(key=lambda r: r.get("created_at", ""))
        return rows[-limit:]

    def current_session(self) -> str:
        """The conversation a new turn belongs to. A session ends when the person clears the chat, so
        turns stay grouped the way they were actually had rather than by an arbitrary time window."""
        sid = self.outbox.kv_get("chat_session", None)
        if not sid:
            sid = ids.make_id("session", self.cfg.device_id, self._now_iso())
            self.outbox.kv_set("chat_session", sid)
        return sid

    def conversations(self, limit: int = 30) -> list[dict]:
        """Past conversations, newest first, each with its first question as a title."""
        turns = [self._plain(r.payload) for r in
                 self.store.scroll(filter={"type": "memory", "kind": "chat",
                                           "device_id": self.cfg.device_id})]
        groups: dict[str, list[dict]] = {}
        for t in turns:
            groups.setdefault(t.get("session") or "older", []).append(t)
        out = []
        for sid, rows in groups.items():
            rows.sort(key=lambda r: r.get("created_at", ""))
            first = re.match(r"^Q:\s*(.+?)(?:\nA:|$)", rows[0].get("note_text", ""), re.S)
            last = re.search(r"\nA:\s*(.*)$", rows[-1].get("note_text", ""), re.S)
            out.append({"session": sid, "turns": len(rows),
                        "title": (first.group(1).strip() if first else "Conversation")[:70],
                        "preview": re.sub(r"\s+", " ", last.group(1)).strip()[:90] if last else "",
                        "started_at": rows[0].get("created_at"),
                        "last_at": rows[-1].get("created_at")})
        out.sort(key=lambda c: c.get("last_at") or "", reverse=True)
        return out[:limit]

    def conversation(self, session: str) -> list[dict]:
        """Every turn of one conversation, oldest first, so it can be reopened."""
        rows = [self._plain(r.payload) | {"id": r.id} for r in
                self.store.scroll(filter={"type": "memory", "kind": "chat",
                                          "session": session, "device_id": self.cfg.device_id})]
        rows.sort(key=lambda r: r.get("created_at", ""))
        return rows

    def forget_chat(self, delete: bool = False) -> int:
        """End the current conversation.

        By default this starts a new one and keeps the old turns, because "clear" on a chat screen means
        "give me a blank page", not "destroy what I said" — and the history is the thing that makes
        follow-up questions work. Passing delete=True really does erase every turn.
        """
        if delete:
            n = self.store.count({"type": "memory", "kind": "chat", "device_id": self.cfg.device_id})
            with self._lock:
                self.store.delete_where({"type": "memory", "kind": "chat",
                                         "device_id": self.cfg.device_id})
            self.outbox.kv_set("chat_session", None)
            self.outbox.log("memory", f"conversation history deleted ({n} turn(s))")
            return n
        n = self.store.count({"type": "memory", "kind": "chat", "session": self.current_session(),
                              "device_id": self.cfg.device_id})
        self.outbox.kv_set("chat_session", None)
        self.outbox.log("memory", f"started a new conversation (kept {n} turn(s))")
        return n

    # How far `ask` may reach for an answer. Stored beside the online flag, so it survives a restart and
    # so the UI reads it back the same way it reads everything else.
    #   local  - this device only. Nothing ever leaves it. The original behaviour.
    #   auto   - the device first; the internet only when the device holds nothing that answers it.
    #   online - the internet on every question, alongside whatever the device holds.
    RETRIEVAL_MODES = ("local", "auto", "online")

    def retrieval_mode(self) -> str:
        mode = self.outbox.kv_get("retrieval_mode", "auto")
        return mode if mode in self.RETRIEVAL_MODES else "auto"

    def set_retrieval_mode(self, mode: str) -> dict:
        if mode not in self.RETRIEVAL_MODES:
            raise ValueError(f"retrieval mode must be one of {self.RETRIEVAL_MODES}")
        self.outbox.kv_set("retrieval_mode", mode)
        return self.retrieval_status()

    def retrieval_status(self) -> dict:
        from edge import online
        return {"mode": self.retrieval_mode(), "online": bool(self.outbox.kv_get("online", True)),
                "search": online.available()}

    def _may_go_online(self, has_local_answer: bool) -> tuple[bool, str]:
        """Whether this question may leave the device, and the reason either way.

        The reason is returned rather than logged because the UI shows it: "answered on the device" and
        "answered using the internet" are different promises to the person asking, and which one they got
        must never be something they have to infer.
        """
        from edge import online
        mode = self.retrieval_mode()
        if mode == "local":
            return False, "set to this device only"
        if not self.outbox.kv_get("online", True):
            return False, "offline"
        if not online.available()["ready"]:
            return False, "no web search configured"
        if mode == "auto" and has_local_answer:
            return False, "this device already holds the answer"
        return True, "online"

    _OVERVIEW_EMPTY = {
        "problems": "Nothing has been flagged on this machine yet. The gate has opened no episode, which "
                    "means every window it has measured sat inside the healthy baseline.",
        "fixed": "No repair has been recorded on this machine yet. Once someone records an action and the "
                 "sensor then sees the machine stay healthy, it will be listed here.",
        "unresolved": "Nothing is outstanding on this machine. No episode is open or waiting on a repair.",
        "known": "This device holds nothing yet. Teach it something, or let it measure the machine, and it "
                 "will remember.",
    }

    def _overview(self, intent: str) -> tuple[list[dict], str]:
        """Answer a question about the whole record from stored episode state.

        Counting is done over what is stored rather than over what retrieval returned: these questions
        are about everything the device holds, and a nearest-neighbour search answers a different
        question. Every sentence is a stored field; nothing here is inferred.
        """
        eps = self.episodes()
        if intent == "fixed":
            picked = [e for e in eps if e.get("action_code") and e.get("outcome") == "worked"]
            head = ("{n} repair(s) on this machine have been recorded as having worked, each confirmed by "
                    "the machine's own sensor data afterwards.")
        elif intent == "unresolved":
            picked = [e for e in eps
                      if e.get("status") != "closed" or not e.get("action_code")
                      or e.get("outcome") in (None, "pending", "failed")]
            head = "{n} thing(s) on this machine are still outstanding."
        elif intent == "problems":
            picked = eps
            head = "{n} episode(s) have been opened on this machine."
        else:                                      # "known": what this device holds, in totals
            picked = eps[:3]
            facts = self.store.count({"type": "memory", "kind": "fact"})
            shared = self.store.count({"type": "shared"})
            pics = self.store.count({"type": "picture", "device_id": self.cfg.device_id})
            worked = sum(1 for e in eps if e.get("outcome") == "worked")
            failed = sum(1 for e in eps if e.get("outcome") == "failed")
            if not (eps or facts or shared or pics):
                return [], self._OVERVIEW_EMPTY["known"]
            head = (f"This is {self.cfg.machine_id} at site {self.cfg.site_id}, watched by device "
                    f"{self.cfg.device_id} on the {self.profile.name} profile. It holds {len(eps)} "
                    f"episode(s) ({worked} repair(s) recorded as worked, {failed} as failed), {facts} "
                    f"thing(s) you taught it, {shared} item(s) shared from other devices and {pics} "
                    f"photo(s). Everything here was searched on this device, with no internet.")
        if not picked and intent != "known":
            return [], self._OVERVIEW_EMPTY[intent]

        used = [{"source": "record", "id": e["episode_id"], "kind": "record", "text": _ep_line(e)}
                for e in picked[:6]]
        if intent == "known":
            body = "\n\n".join(f"{c['text']} [E{i}]" for i, c in enumerate(used, 1))
            return used, head + ("\n\nThe most recent:\n\n" + body if used else "")
        lead = head.format(n=len(picked))
        if len(picked) > len(used):
            lead += f" The {len(used)} most recent:"
        return used, lead + "\n\n" + "\n\n".join(f"{c['text']} [E{i}]" for i, c in enumerate(used, 1))

    @staticmethod
    def _as_question(q: str) -> str:
        """Phrase a bare topic ("capital of tamilnadu") as the question the reader was trained on. The same fact
        scores about 6 as a fragment and about 11 as "What is the capital of ...?"."""
        t = q.strip().rstrip("?.! ")
        if re.match(r"(?i)^(what|who|whom|whose|which|when|where|why|how|is|are|was|were|do|does|did|can|could|"
                    r"will|would|should|has|have)", t):
            return t + "?"
        return f"What is the {t}?" if re.match(r"(?i)^(capital|population|meaning|definition|full form|currency|"
                                               r"area|height|length|speed|symbol|formula)", t) else f"What is {t}?"

    QA_MARGIN = 6.0     # how much better than "no answer" a span must score (calibrated in bench/ask_qa.py, D49)

    def _read_wiki(self, q: str, qw: set[str], cands: list[dict], qa) -> list[dict]:
        """The Wikipedia/technical lead that really answers `q`, read by the offline extractive model; [] if none does.
        Title matches are read first; at most 8 candidates are read (about 80 ms each)."""
        named = [c for c in cands if c.get("title_hit")]
        if _COMPARE.search(q):
            # "PLC vs DCS": the library holds each definition, not a comparison. Showing both, labelled as what they
            # are, is true; inventing a side-by-side would not be.
            seen, out = set(), []
            for c in named:
                if c.get("title") not in seen:
                    seen.add(c.get("title"))
                    out.append(c)
            if len(out) >= 2:
                for c in out[:2]:
                    c["compare"] = True
                return out[:2]
        if _OPEN_ENDED.match(q):
            return []                              # nothing to quote for a design / calculation / diagnosis
        order = sorted(cands, key=lambda c: (-bool(c.get("title_hit")), -len(c["overlap"])))[:8]
        best = None
        rest = qw - _SOFT_WORDS
        for c in order:
            # the question names this entry and nothing else ("what is a PLC", "how does an encoder work"): the lead IS
            # the answer. The entry may be known by several names, so every alias counts.
            names = [c.get("title") or ""] + list(c.get("aliases") or [])
            if c.get("title_hit") and any(_covered(rest, n) == rest for n in names) and rest:
                return [c]
            try:
                r = qa.answer(self._as_question(q), c["text"])
            except Exception:                      # a broken reader must not cost the answer
                continue
            if r and r["margin"] >= self.QA_MARGIN:
                # The reader found a span, but is it the right ENTITY? "first prime minister of India" matched Tunisia's
                # article. Every real word of the question (not filler, not the question verb) must occur in the passage
                # or in the entry's names.
                seen_text = " ".join([c["text"], c.get("title") or ""] + list(c.get("aliases") or []))
                if (qw - _SOFT_WORDS - _ASK_VERBS) - _covered(qw, seen_text):
                    continue
            if r and r["margin"] >= self.QA_MARGIN and (best is None or r["margin"] > best[0]):
                best = (r["margin"], c, r)
        if not best:
            return []
        _, c, r = best
        c["reader_answer"], c["answer_sentence"], c["margin"] = r["text"], r["sentence"], r["margin"]
        return [c]

    def _session_id(self, session: str | None) -> str:
        """The conversation this turn belongs to: the one the caller names, else the device's current one."""
        if session and re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}", session):
            return session
        return self.current_session()

    def _plain_reply(self, q: str, answer: str, mode: str, sid: str, lang: str, trace: dict | None = None) -> dict:
        """A turn answered without retrieval (arithmetic, casual talk, a clarifying question)."""
        out = {"question": q, "answer": answer, "grounded": mode != "clarify", "mode": mode, "model": None, "llm_ms": 0,
               "used": [],
               "sources": [], "web_reason": mode, "memory_failed": [], "retrieval_mode": self.retrieval_mode(),
               "language": lang, "translated": False, "answer_en": None, "pictures": [], "retrieval": None,
               "needs_internet": False, "latency_ms": 0, "session": sid, "trace": {"session": sid, **(trace or {})}}
        if mode == "clarify":      # the person's next message may answer it, so the question and the reply are kept
            self._log_turn(q, answer, sid, [], q)
        return out

    def _log_turn(self, q: str, answer: str, sid: str, topics: list[str], resolved: str) -> None:
        try:
            self.remember(f"Q: {q}\nA: {answer}", kind="chat",
                          meta={"session": sid, "topics": topics, "resolved": resolved})
        except Exception:
            pass                                   # a store that cannot save the chat must not lose the answer

    def ask(self, text: str, llm=None, use_fleet: bool = True, limit: int = 5, session: str | None = None) -> dict:
        """One conversational turn.

        The pipeline is one line of decisions, each made once (docs/DECISIONS.md D50):
          1. arithmetic -> exact answer, no model
          2. edge/converse.py understands the message against THIS conversation: casual talk gets a conversational reply
             (no retrieval); a request for a topic becomes a canonical question; a follow-up has its reference replaced by
             the current topic; an unclear reference gets a clarifying question
          3. retrieval over everything this device can reach (taught notes, sensor records, the fleet mirror, the offline
             library), each hit filtered by whether it actually ANSWERS (relevance + the extractive reader)
          4. the answer, with its sources named separately; the turn and its topic are stored for the next follow-up

        `session` is the conversation this turn belongs to. The UI sends it; without one the device's current session is used.
        """
        from edge import converse, rag             # local import: rag pulls in the optional LLM stack
        q = (text or "").strip()
        if not q:
            raise ValueError("ask needs a question")
        sid = self._session_id(session)
        calc = _arithmetic(q)
        if calc:
            return self._plain_reply(q, f"{calc[0]} = {calc[1]}", "calculated", sid, "en")
        # casual talk is decided on the raw text (it must work in Hindi too) and never touches retrieval
        talk = converse.casual(q)
        if talk:
            return self._plain_reply(q, talk[1], "smalltalk", sid, "hi" if re.search(r"[ऀ-ॿ]", q) else "en",
                                     trace={"message_type": "casual", "kind": talk[0]})
        # Everything below is English: retrieval, the grounding check, the evidence texts. A question in
        # Hindi is translated once here, answered by that same pipeline, and the answer is translated back
        # at the end, so the person is answered in the language they asked in.
        from shared import translate
        original_q, lang, translated = q, translate.detect(q), False
        if lang == "hi" and translate.available("hi"):
            try:
                q, translated = translate.to_english(q, "hi") or q, True
            except Exception:                      # a missing or broken model must not cost the answer
                q = original_q
        t0 = time.perf_counter()
        history = self.recent_chat(8, session=sid)
        # "what problems have you seen?", "what do you know about me?" ask about this device's own record as a whole. They
        # are complete questions with no subject to resolve, so they bypass reference resolution (D46: overview intents).
        if not _identifiers(q) and _overview_intent(q):
            und = converse.Understanding("knowledge", resolved=q, topics=[], notes={"followup": False, "overview": True})
        else:
            und = converse.understand(q, history)
        if und.kind in ("casual", "clarify"):
            return self._plain_reply(original_q, und.reply, "smalltalk" if und.kind == "casual" else "clarify", sid,
                                     "en", trace={"message_type": und.kind, **und.notes})
        follow_up = und.kind == "followup"
        # the self-contained question: "what are its inputs?" has become "what are system's inputs?" before anything is
        # searched. A standalone question is left exactly as asked, so an old topic never leaks into a new one.
        asked_as, q = q, und.resolved or q
        ctx_text, ctx_words, ctx_ids = "", set(), set()
        query = q
        qw = _content_words(q)

        cands: list[dict] = []
        # A failing store (a Qdrant error, a locked shard) must cost this answer its device memory, not the
        # whole turn: each search leg is isolated, and the answer says which leg was lost so a missing
        # result is never mistaken for "this device holds nothing about that".
        memory_failed: list[str] = []

        def _safe(label, fn, *args, **kw):
            try:
                return fn(*args, **kw)
            except Exception:
                memory_failed.append(label)
                return []
        # only facts the user taught are evidence; past turns are conversational context, and quoting
        # them back would let the device cite its own earlier answer as if it were a source
        for m in _safe("memory", self.recall, query, limit, kind="fact"):
            cands.append({"source": "memory", "id": m["id"], "kind": "fact",
                          "text": m.get("note_text", "")})
        # What it has already looked up stays on the device and is searched like anything else. This is
        # the whole point of the product: a machine that reached the internet once should still know the
        # answer with the network off, instead of saying "this device is offline" about a thing it read
        # last week.
        for m in _safe("learned", self.recall, query, limit, kind="learned"):
            cands.append({"source": "learned", "id": m["id"], "kind": "learned",
                          "text": m.get("note_text", ""), "url": m.get("url"),
                          "title": m.get("title")})
        for s in _safe("shared", self.recall_shared, query, limit):
            cands.append({"source": "shared", "id": s["id"], "kind": "shared",
                          "text": s.get("note_text", ""), "from": s.get("author_device")})
        for rf in _safe("offline library", self.recall_reference, query, limit):
            cands.append({"source": "reference", "id": rf["id"], "kind": "reference",
                          "text": rf.get("note_text", ""), "title": rf.get("title"),
                          "sources": rf.get("sources") or [], "title_hit": bool(rf.get("title_hit")),
                          "topic": rf.get("topic"), "aliases": rf.get("aliases") or []})
        try:
            res = self.search(text=query, use_fleet=use_fleet, limit=limit)
        except Exception:                          # memory-only questions must work with no baseline yet
            res = {"local": [], "fleet": [], "latency_ms": 0}
        for r in res.get("local", []):
            cands.append({"source": "record", "id": r["id"], "kind": "record",
                          "text": rag._local_item(r["episode"])})
        for r in res.get("fleet", []):
            cands.append({"source": "fleet", "id": r["id"], "kind": "fleet",
                          "text": rag._fleet_item(r["case"])})

        # relevance by word overlap: cheap, explainable, and it does not depend on a score threshold
        # (RRF scores are rank-based and not comparable across queries, so they cannot be thresholded)
        # a question that names its own thing keeps to it: "plot 91" must never inherit "plot 44"
        own_ids = _identifiers(q)
        q_ids = own_ids or (ctx_ids if follow_up else set())
        for c in cands:
            c["overlap"] = sorted(_covered(qw, c["text"], plural=c["source"] in ("reference", "learned")))
            if c.get("title_hit"):               # the entry was asked for BY NAME, and the name may not be in its text
                c["overlap"] = sorted(set(c["overlap"]) | _covered(qw, " ".join([c.get("title") or ""] + list(c.get("aliases") or []))))
            c["ids"] = sorted(q_ids & _identifiers(c["text"]))
        # when the question names a specific thing, only records naming the same thing count
        # One shared word is enough for a note somebody typed in, but not for the offline library: with tens
        # of thousands of articles, some article always shares a word with any question, so "who painted the
        # ceiling of the zxqv chapel" matched Dante and the Sistine Chapel and got answered from them. A
        # library entry has to cover most of what was asked (at least 60 %, and at least two words).
        from shared import qa as _qa
        qa_on = _qa.available()                    # the offline reader that decides whether a passage answers

        # The topic(s) the understanding layer found. A device record, fleet case, shared note or cached web page must be about
        # the topic: "where is DNA found" shares the word "found" with a bearing episode, which says nothing about DNA.
        topic_stems = {_stem(t.lower()) for tp in und.topics for t in re.findall(r"[A-Za-z0-9]+", tp)
                       if t.lower() not in _GENERIC and len(t) > 1}

        def _on_topic(c: dict) -> bool:
            if not topic_stems:
                return True
            hay = (c.get("title") or "") + " " + (c["text"][:160] if c["source"] == "learned" else c["text"])
            return bool(topic_stems & {_stem(t) for t in re.findall(r"[a-z0-9]+", hay.lower())})

        def _relevant(c: dict) -> bool:
            if q_ids:
                return bool(c["ids"])
            if c["source"] != "reference" and not _on_topic(c):
                return False
            # Text nobody on this tenant wrote - the library, a page cached from the web - has to cover most of the
            # question. Notes people typed in (here or on another device of the fleet) and this device's own sensor
            # records may match on a single word. Without this a cached page about the Great Wall of China answered
            # "how many states does India have?" because both contain "states".
            if c["source"] in ("reference", "learned"):
                if len(qw) < 2:
                    return bool(c["overlap"])
                # A curated fact answers one narrow question, so it must cover EVERY meaningful word of it (filler
                # such as "city" or "name" excepted): "national animal of India" matched the fact about the capital
                # of the National Capital Territory on two of its three words and answered with that. An article
                # whose title is what was asked can be looser; anything else needs three words in four.
                if c.get("topic") == "general knowledge":
                    return len(qw - _SOFT_WORDS - set(c["overlap"])) == 0
                if qa_on and c.get("topic") in _ENCYCLOPEDIC:
                    return bool(c["overlap"])      # topic is enough to be READ; the reader decides whether it answers
                ratio = 0.6 if c.get("title_hit") else 0.75
                return len(c["overlap"]) >= max(2, math.ceil(len(qw) * ratio - 1e-9))
            return bool(set(c["overlap"]) - _GENERIC)
        used = [c for c in cands if _relevant(c)]
        # A record that IS the thing asked about must outrank one that merely mentions it: the entry for
        # P0305 says "same family as P0301", so asking about P0301 otherwise answers with P0305.
        used.sort(key=lambda c: (-_names_it(c["text"], c["ids"]), -len(c["ids"]), -len(c["overlap"])))
        # library entries are chosen as a group (title match first, best coverage next, no near-duplicates)
        lib = [c for c in used if c["source"] == "reference"]
        if lib:
            wiki_all = [c for c in lib if c.get("topic") in _ENCYCLOPEDIC] if qa_on else []
            keep = {id(c) for c in _trim_reference([c for c in lib if c not in wiki_all], len(qw))}
            # A Wikipedia lead that shares the question's words is about the same TOPIC; that is not the same as
            # answering it ("largest mammal" -> the article on elephants, "father of computers" -> a biography of
            # Turing). So each candidate is READ by an extractive question-answering model (shared/qa.py): it returns
            # the sentence that answers, made of the passage's own words, or declines. A curated fact, when there is
            # one, is the answer and nothing is read. Only a question that IS a title ("what is DNA") skips the
            # reading, because there the lead is the answer by definition.
            if wiki_all and not any(c.get("topic") == "general knowledge" for c in lib if id(c) in keep):
                keep |= {id(c) for c in self._read_wiki(q, qw, wiki_all, _qa)}
            used = [c for c in used if c["source"] != "reference" or id(c) in keep]
        used = used[:6]
        for i, c in enumerate(used, 1):
            c["key"] = f"E{i}"

        # "What problems have you seen?" asks about the record as a whole and names nothing in it, so
        # word overlap finds nothing however much the device holds. A question that does name a specific
        # thing ("has plot 91 been fixed?") keeps the retrieval path: there the overview would be wrong.
        intent = None if own_ids else _overview_intent(q)
        if intent:
            used, overview_answer = self._overview(intent)
            for i, c in enumerate(used, 1):
                c["key"], c["overlap"], c["ids"] = f"E{i}", [], []

        # ---- the wider world ---------------------------------------------------------------------------
        # An overview question is about this device's own record, so it never leaves the device however
        # the retrieval mode is set: there is no answer to "what have you seen?" on the internet.
        web_reason = "a question about this device"
        if not intent:
            may, web_reason = self._may_go_online(has_local_answer=bool(used))
            if may:
                from edge import online
                hits = online.search(q, limit=max(3, min(limit, online.MAX_RESULTS)))
                # A search engine always returns something, exactly like the vector store does, so web
                # hits face the same relevance test as everything else: a result sharing no word with
                # the question is not evidence. Asking about the president of France otherwise cited
                # "Norfolk State University", which is worse than citing nothing.
                web = [w for w in online.as_evidence(hits) if qw & _content_words(w["text"])]
                for w in web:                      # keep it, so the next answer needs no network
                    try:
                        self.remember(w["text"][:self.MEMORY_MAX_CHARS], kind="learned",
                                      meta={"url": w["url"], "title": w["title"], "learned_for": q})
                    except Exception:
                        pass                       # a cache that fails must never cost the live answer
                if web:
                    # a page already on the device and a fresh fetch of it are one source, not two
                    fresh = {w["url"] for w in web}
                    used = [c for c in used if c.get("url") not in fresh]
                    # web evidence sits after local evidence and shares the one [E1..] sequence, so the
                    # grounding check in edge/rag.py keeps working unchanged
                    used = used + web
                    for i, c in enumerate(used, 1):
                        c["key"] = f"E{i}"
                        c.setdefault("overlap", [])
                        c.setdefault("ids", [])
                elif not online.reachable():
                    # the switch says online but nothing answered: that is offline, whatever the switch says
                    web_reason = "offline"
                else:
                    web_reason = "the internet returned nothing for this"

        if intent:
            out = {"answer": overview_answer, "grounded": bool(used), "mode": "overview",
                   "model": None, "llm_ms": 0,
                   "used": [{"key": c["key"], "source": c["source"], "origin": _origin(c["source"]),
                             "id": c["id"], "text": c["text"],
                             "title": None, "sources": [], "matched": []} for c in used]}
        elif not used:
            # Say which of the two it is. "Nothing matched" and "this device could never know that" feel
            # identical from the outside but need different things from the person: one is a search that
            # missed, the other needs teaching or a source this device does not have.
            # retrieval always returns candidates, so "did it return anything" says nothing. The signal
            # is whether a single stored record shares even one word with the question: none at all means
            # the subject is outside this device's world, rather than a search that just missed.
            # the library always shares a word with something, so only the device's own records count as a near miss
            searched_anything = any(c["overlap"] for c in cands if c["source"] != "reference")
            # Saying "that would need an internet connection" when the device HAS one, and simply was not
            # allowed or able to use it, is the complaint this whole path exists to answer. Each reason
            # gets its own sentence, because each one needs something different from the person.
            why = {
                "offline": "This needs an internet connection, and this device is offline, so it cannot look it "
                           "up. Turn the network back on and ask again, or teach it with “Teach it something”.",
                "set to this device only": "Ask is set to search this device only. Settings → Ask will let "
                                           "it use the internet as well.",
                "no web search configured": "No web search is configured on this device, so it cannot look "
                                            "it up. Set EDGE_SEARCH_PROVIDER, or teach it.",
                "the internet returned nothing for this": "I searched the internet too and found nothing "
                                                          "that answers it.",
            }.get(web_reason, "")
            needs_net = web_reason in ("offline", "set to this device only", "no web search configured")
            if needs_net:
                # The one sentence a person needs: this device does not know it and the internet could. The
                # hints that follow say how to make the internet available when that is a setting, not a state.
                answer = NEEDS_INTERNET + {"set to this device only": " Ask is set to search this device only. "
                                           "Settings → Ask will let it use the internet as well.",
                                           "no web search configured": " No web search is configured on this "
                                           "device. Set EDGE_SEARCH_PROVIDER, or teach it."}.get(web_reason, "")
            elif searched_anything:
                answer = ("I found nothing close enough to answer that. This device does hold records that "
                          "touch on some of those words, but none of them answer the question. Try naming "
                          "the part, code or symptom directly, or teach it.")
            else:
                answer = "Nothing on this device relates to that. " + (
                    why or "I searched the internet too and found nothing that answers it.")
            out = {"answer": answer, "grounded": False, "mode": "no_evidence", "used": [],
                   "model": None, "llm_ms": 0, "web_reason": web_reason,
                   "needs_internet": needs_net, "sources": []}
            # Nothing found anywhere. The local model may answer ONLY a short, simple general-knowledge question
            # (never anything about machines, faults, repairs or this device), only after agreeing with itself
            # across three runs, and the answer is labelled unverified because no source backs it. When it is
            # not sure it says it does not know. This is deliberately last: every sourced answer comes first.
            if llm is not None and getattr(llm, "available", False) and _simple_general(q):
                lt = time.perf_counter()
                try:
                    g = rag.general_answer(llm, q, samples=1 if rag.is_big_model() else 2)
                except Exception:                  # the optional model must never break the answer
                    g = None
                ms = round((time.perf_counter() - lt) * 1000)
                if g:
                    out = {"answer": f"{rag.UNVERIFIED_PREFIX} {g}", "grounded": False, "mode": "model_unverified",
                           "unverified": True, "label": rag.UNVERIFIED_LABEL, "model": rag.MODEL_NAME, "llm_ms": ms,
                           "used": [], "web_reason": web_reason, "needs_internet": False, "sources": []}
                else:
                    if not out.get("needs_internet"):
                        out["answer"] = "I don't know. " + out["answer"]
                    out["model_unsure"] = True
        else:
            kinds = {c["source"] for c in used}
            if kinds == {"reference"}:
                used = used[:2]                    # the library is quoted, so show the best two, not six
            texts = {c["key"]: c["text"] for c in used}
            # quoting the source verbatim is a fine answer, not a failure: the reference packs are already
            # written as prose and cite where they came from, so this says so plainly instead of apologising
            # The lead has to match where the lines actually came from. "From what this device holds"
            # over a block of web snippets is the one sentence in the whole product that would be a lie.
            lead = ("The library holds each definition, not a side-by-side comparison:"
                    if any(c.get("compare") for c in used) else
                    "From the internet:" if kinds == {"web"} else
                    "From this device, and from the internet:" if "web" in kinds else
                    "" if kinds == {"reference"} else            # the badge under the answer names the library
                    "From the offline library, and from this device:" if "reference" in kinds else
                    "From what this device looked up earlier:" if kinds == {"learned"} else
                    "From what this device holds:")

            def _body(c: dict) -> str:
                # library articles are stored as "Title: lead"; the title is already the heading of the source
                # card, so repeating it in the sentence read as noise ("Tamil Nadu: Tamil Nadu is a state ...")
                if c.get("answer_sentence"):
                    return c["answer_sentence"]
                t, ttl = c["text"], c.get("title") or ""
                return t[len(ttl) + 2:] if ttl and c["source"] == "reference" and t.startswith(ttl + ": ") else t
            template = (lead + "\n\n" if lead else "") + "\n\n".join(f"{_body(c)} [{c['key']}]" for c in used)
            # The offline library is third-party text. A paraphrase of it can be wrong while still carrying a
            # citation, so library-only answers are always the passage itself, word for word.
            if llm is not None and getattr(llm, "available", False) and kinds != {"reference"}:
                # the recent turns go in as conversation, never as citable evidence, so a pronoun can be
                # resolved without the model being able to cite its own earlier answer back as a source
                convo = ""
                if follow_up and history:
                    convo = "Earlier in this conversation:\n" + "\n".join(
                        h.get("note_text", "") for h in history[-CHAT_CONTEXT_TURNS:]) + "\n\n"
                # "Evidence:" exactly matches rag.EXAMPLE_USER: the one-shot example is what teaches the
                # model to append [E1], and heading the block differently was enough to lose the citations,
                # which then cost every sentence at the grounding check
                prompt = (convo + "Evidence:\n" + "\n".join(f"[{k}] {v}" for k, v in texts.items())
                          + f"\nQuestion: {q}"
                          + "\nAnswer using only the evidence above. End EVERY sentence with its evidence "
                            "id, like [E1].")
                lt = time.perf_counter()
                try:
                    raw = llm.complete(prompt)
                    if len(texts) == 1:
                        raw = _attach_lone_citation(raw, next(iter(texts)))
                    kept, _ = rag.check_output(raw, set(texts), texts)
                    # a sentence that is nothing but its citation makes no claim, and on its own it
                    # rendered as a bubble containing the single word "E1"
                    kept = [k for k in kept if re.sub(r"\[E\d+\]", "", k).strip(" .")]
                    # a model sentence may only use words that are in the evidence it cites (or in the
                    # question); one that brings its own facts is dropped and the evidence is quoted instead
                    kept = [k for k in kept if _supported(k, texts, q)]
                    ms = round((time.perf_counter() - lt) * 1000)
                    out = ({"answer": " ".join(kept), "grounded": True, "mode": "llm",
                            "model": rag.MODEL_NAME, "llm_ms": ms} if kept else
                           {"answer": template, "grounded": True, "mode": "quoted",
                            "model": rag.MODEL_NAME, "llm_ms": ms})
                except Exception:                  # the optional model must never break the answer
                    out = {"answer": template, "grounded": True, "mode": "quoted", "model": None, "llm_ms": 0}
            else:
                out = {"answer": template, "grounded": True, "mode": "quoted", "model": None, "llm_ms": 0}
            # Where the answer actually came from. The person is promised different things by "this
            # device knows" and "the internet says", so the answer carries which it was rather than
            # leaving the UI to guess from the evidence list.
            out["sources"] = sorted({_origin(c["source"]) for c in used})
            # answered offline from something it looked up before: worth saying, because it is the
            # difference between "this device is useless without a network" and "it remembers"
            out["from_learned"] = any(c["source"] == "learned" for c in used)
            out["used"] = [{"key": c["key"], "source": c["source"], "origin": _origin(c["source"]),
                            "id": c["id"], "text": c["text"], "url": c.get("url"),
                            "title": c.get("title"), "sources": c.get("sources") or [],
                            "matched": c["overlap"]} for c in used]

        # Pictures ride alongside the answer, never inside it. A photograph supports no sentence — nothing
        # here reads one — so it is offered as "you also hold these" and a person decides what it shows.
        pics = []
        try:
            # Loading the image model for every question cost hundreds of milliseconds on any device holding a
            # photo, and only a picture whose note shares a word with the question can be shown anyway - so the
            # notes are compared first and the image model is only used when one could match.
            if self.store.count({"type": "picture", "device_id": self.cfg.device_id}) and any(
                    qw & _content_words(p.get("note_text", "")) for p in self.pictures(100)):
                # nearest-neighbour always returns something, so the top hits include pictures with nothing
                # to do with the question. Only those whose note actually shares words with it are shown:
                # the answer presents these as matching, and a photograph offered under a question it has
                # no bearing on is worse than showing none. Searching pictures directly still uses the
                # full CLIP ranking, because there the person is deliberately browsing images.
                pics = [{"id": p["id"], "note": p.get("note_text", ""), "filename": p.get("filename"),
                         "score": p.get("score")}
                        for p in self.recall_images(text=q, limit=4)
                        if qw & _content_words(p.get("note_text", ""))][:3]
        except Exception:
            pics = []                              # a missing CLIP model must never break a text answer

        out.setdefault("sources", ["device"] if out.get("used") else [])
        out.setdefault("web_reason", web_reason)
        out["memory_failed"] = memory_failed       # search legs that errored: the answer may be missing device evidence
        if memory_failed and not out.get("used"):
            out["answer"] = ("Searching " + " and ".join(memory_failed) + " on this device failed, so this "
                             "answer may be missing what the device holds. ") + out["answer"]
        out["retrieval_mode"] = self.retrieval_mode()

        answer_en = None
        if translated:
            answer_en = out["answer"]
            try:
                out["answer"] = translate.from_english(answer_en, "hi")
            except Exception:
                translated, answer_en = False, None   # say it in English rather than not at all

        self._log_turn(original_q, out["answer"], sid, und.topics, q)
        trace = {"session": sid, "message_type": und.kind, "original": original_q, "asked_as": asked_as,
                 "resolved": q, "topics": und.topics, "context_turns": len(history), **und.notes,
                 "retrieval_query": query, "mode": out.get("mode"), "sources": out.get("sources"),
                 "candidates": [{"source": c["source"], "title": c.get("title"), "overlap": c.get("overlap")}
                                for c in cands[:10]],
                 "selected": [{"key": c.get("key"), "source": c["source"], "title": c.get("title")} for c in used]}
        if os.environ.get("EDGE_ASK_TRACE") == "1":
            try:                                   # logging must never be able to cost a person their answer
                print("[ask-trace] " + json.dumps(trace, ensure_ascii=True, default=str), flush=True)
            except Exception:
                pass
        return {"question": original_q, **out, "language": lang, "translated": translated,
                "answer_en": answer_en, "pictures": pics, "session": sid, "trace": trace,
                "retrieval": self._retrieval_report(cands, used, res, follow_up, query),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    # what Qdrant was actually asked, for the Qdrant screen. Reporting only — it recomputes nothing.
    _SOURCE_LABELS = {
        "memory":    ("Things you taught it", {"type": "memory", "kind": "fact"}),
        "shared":    ("Shared with this device", {"type": "shared"}),
        "reference": ("Built-in reference", {"type": "reference"}),
        "record":    ("What the sensor noticed", {"type": "episode"}),
        "fleet":     ("Fleet mirror (from the cloud)", None),
    }

    def _retrieval_report(self, cands: list[dict], used: list[dict], res: dict,
                          follow_up: bool, query: str) -> dict:
        by_source: dict[str, int] = {}
        for c in cands:
            by_source[c["source"]] = by_source.get(c["source"], 0) + 1
        kept = {c["source"] for c in used}

        searched = []
        for src, (label, flt) in self._SOURCE_LABELS.items():
            try:
                held = (self.mirror.count() if src == "fleet"
                        else self.store.count(dict(flt, device_id=self.cfg.device_id)
                                              if "device_id" not in (flt or {}) else flt))
            except Exception:
                held = None
            searched.append({"source": src, "label": label, "stored": held,
                             "returned": by_source.get(src, 0),
                             "kept": sum(1 for c in used if c["source"] == src)})

        return {
            "query_sent": query,
            "follow_up": follow_up,
            "where": "Qdrant Edge, on this device" + (" + fleet mirror" if res.get("fleet") else ""),
            "searched": searched,
            "vectors": [
                {"name": "note", "kind": "dense text", "dim": self.store.meta.get("note_dim"),
                 "model": self.embedder.name, "used": True},
                {"name": "note_bm25", "kind": "sparse keyword (BM25)", "dim": None,
                 "model": "built into Qdrant Edge", "used": True},
                {"name": "vib", "kind": "vibration fingerprint", "dim": self.store.meta.get("vib_dim"),
                 "model": self.profile.fp_version, "used": bool(res.get("local"))},
                {"name": "image", "kind": "picture (CLIP)", "dim": self.store.meta.get("image_dim"),
                 "model": "Qdrant/clip-ViT-B-32", "used": bool(self.store.meta.get("image_vector"))},
            ],
            "fusion": "reciprocal rank fusion of every leg, in one Qdrant Edge request",
            "kept_sources": sorted(kept),
            "total_points": sum(s["stored"] or 0 for s in searched),
            "search_ms": round(res.get("latency_ms", 0), 2),
        }

    def search(self, text: str | None = None, episode_id: str | None = None, use_fleet: bool = True,
               limit: int = 5) -> dict:
        """Hybrid search. Local: this machine's episodes by fingerprint + note (dense) + note (BM25), fused with RRF.
        Fleet (K2 decision): filtered by component and the episode's fault class (technician-confirmed, else the
        physics hint), ranked by text; the fingerprint is only a low-weight tie-break across machines."""
        t0 = time.perf_counter()
        vib, fc, src = None, None, None
        if episode_id:
            rec = self.store.get(episode_id, with_vectors=True)
            if rec is None:
                raise KeyError(episode_id)
            vib = rec.vectors.get("vib")
            fc = rec.payload.get("fault_class") or (rec.payload.get("fault_hint") or {}).get("fault_class")
            src = "technician" if rec.payload.get("fault_class") else "physics_hint"
            if not text:
                text = self._doc_text(rec.payload)
        if not text and vib is None:
            raise ValueError("give a text query or an episode")
        note = self.embedder.embed_query(text) if text else None
        local = self.store.search(vib=vib, note=note, text=text, limit=limit + 1, explain=True,
                                  filter={"type": "episode", "machine_id": self.cfg.machine_id})
        local = [h for h in local if h.id != episode_id][:limit]
        fleet, fleet_filter, fleet_error = [], None, None
        try:
            if use_fleet and self.mirror.count():
                fleet_filter = {"component": self.component, "!status": "retracted"}
                if fc and fc != "unknown":
                    fleet_filter["fault_class"] = fc
                # the mirror's text vectors come from the cloud's model; a query vector from another model would
                # compare meaningless numbers, so that leg is dropped and BM25 + fingerprint carry the fleet search
                fleet_model = self.outbox.kv_get("fleet_text_model", FLEET_TEXT_MODEL)   # announced by the cloud
                same_model = self.embedder.name == fleet_model
                fleet = self.mirror.search(vib=vib, note=note if same_model else None, text=text, filter=fleet_filter,
                                           limit=limit, weights={"vib": 0.25}, explain=True)
                if not same_model:
                    fleet_error = (f"fleet dense-text leg skipped: this device embeds with {self.embedder.name}, "
                                   f"the fleet with {fleet_model}; BM25 + fingerprint used")
        except Exception as e:              # a broken mirror must never take local memory down with it
            fleet, fleet_error = [], f"fleet mirror unavailable ({type(e).__name__}); showing local memory only"
        ms = (time.perf_counter() - t0) * 1000
        self.search_ms.append(ms)
        slim = lambda p: {k: v for k, v in p.items() if k not in ("hint_votes",)}
        return {"query": {"text": text, "episode_id": episode_id, "fault_class": fc, "fault_class_source": src,
                          "fleet_filter": fleet_filter},
                "local": [{"id": h.id, "rrf": round(h.score, 4), "legs": h.legs, "episode": slim(h.payload),
                           "feedback": self.feedback_for(h.id, "local")} for h in local],
                "fleet": [{"id": h.id, "rrf": round(h.score, 4), "legs": h.legs, "case": h.payload,
                           "feedback": self.feedback_for(h.id, "fleet")} for h in fleet],
                "fleet_error": fleet_error, "latency_ms": round(ms, 2)}

    def set_share_state(self, event_id: str, state: str, episode_id: str) -> None:
        with self._lock:
            if self.store.get(episode_id) is not None:
                self._set(episode_id, share_state=state)

    def stats(self) -> dict:
        g = list(self.gate_ms)
        s = list(self.search_ms)
        pct = lambda xs, q: round(float(np.percentile(xs, q)), 3) if xs else None
        return {
            "device_id": self.cfg.device_id, "site_id": self.cfg.site_id, "machine_id": self.cfg.machine_id,
            "machine_class": self.cfg.machine_class, "component": self.component, "profile": self.profile.describe(),
            "baseline_capture": self.capture_state(), "last_diagnosis": self.last_diagnosis,
            "risk_hint": getattr(self, "risk_hint", None), "note_protection": self.note_protection,
            "machine_card": self.card.to_dict() if self.card else None,
            "fleet_hint_model": self._model_summary(), "hold_days": self.cfg.hold_days,
            "clock": self.clock_status(),
            "local_detector": ({k: v for k, v in (self.outbox.kv_get("local_detector") or {}).items()
                                if k in ("trained_on", "cross_validated", "trained_at")} or None)
            if self.profile.learned_detector else None,
            "baseline_ready": self.gate is not None, "gate": self.gate.cfg.to_dict() if self.gate else None,
            "windows": dict(self.counters), "last_gate": self.last_gate, "recent": list(self.recent),
            "gate_ms": {"p50": pct(g, 50), "p95": pct(g, 95), "n": len(g)},
            "search_ms": {"p50": pct(s, 50), "p95": pct(s, 95), "n": len(s)},
            "local_points": self.store.facet("type"), "episodes_by_status": self.store.facet("status", filter={"type": "episode"}),
            "share_states": self.store.facet("share_state", filter={"type": "episode"}),
            "mirror_cases": self.mirror.count(), "outbox": self.outbox.counts(),
            "raw_bytes_kept_local": int(self.counters["windows"] * 2048 * 4),
            "embedder": self.embedder.name,
        }

    def _model_summary(self) -> dict | None:
        m = self.fleet_model()
        return None if m is None else {k: m.get(k) for k in ("classes", "trained_on", "unseen_device_accuracy",
                                                             "unseen_device_cases", "trained_at")}

    def close(self) -> None:
        with self._lock:
            self.store.close()
            self.mirror.close()
            self.outbox.close()
