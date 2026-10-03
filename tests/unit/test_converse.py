"""The conversation-understanding layer (edge/converse.py), tested on language structure, not on topics.

Every group below uses SEVERAL unrelated subjects and many phrasings, so a passing test shows the layer generalises: the module
contains no subject and no list of whole phrases (the last test checks the source for that).
"""
import pathlib
import re

import pytest

from edge import converse as C


def run(messages):
    """Play a conversation the way Device.ask does: the topics of each answered turn are stored for the next one."""
    hist, out = [], []
    for m in messages:
        u = C.understand(m, hist)
        out.append(u)
        if u.kind in ("knowledge", "followup"):
            hist.append({"topics": u.topics})
    return out


# ---- A. casual conversation never becomes a search ------------------------------------------------------------------
@pytest.mark.parametrize("msg", [
    "Hi", "Hello", "Hey", "hii", "Good morning", "good evening", "How are you?", "How r u", "What's up?", "whats up",
    "Thanks", "Thank you", "thank you so much", "thx", "You're welcome", "Bye", "goodbye", "See you later", "Good night",
    "ok cool", "hello there!", "hey buddy", "Hi there, how are you?"])
def test_casual_talk_is_answered_conversationally(msg):
    u = C.understand(msg, [])
    assert u.kind == "casual" and u.reply and "Needs internet" not in u.reply


def test_casual_replies_fit_the_message():
    assert C.understand("Good morning", []).reply.startswith("Good morning")
    assert "welcome" in C.understand("Thanks", []).reply.lower()
    assert "goodbye" in C.understand("Bye", []).reply.lower()
    assert C.understand("Hi", []).reply == "Hi! How can I help?"


@pytest.mark.parametrize("msg", ["hi what is a plc", "thanks for the dna info, what about rna", "hello world program",
                                 "history of rome", "ok so what is gravity"])
def test_a_greeting_followed_by_a_real_question_is_a_question(msg):
    assert C.understand(msg, []).kind != "casual"


def test_hindi_greeting_gets_a_hindi_reply():
    u = C.understand("नमस्ते", [])
    assert u.kind == "casual" and "नमस्ते" in u.reply


# ---- B/F. the same request in many wordings means the same thing, for any topic -------------------------------------
TOPICS = ["DNA", "photosynthesis", "a CPU", "HTTP", "gravity", "a database", "the Roman Empire", "quantum entanglement"]
WORDINGS = ["What is {x}?", "Tell me about {x}.", "Explain {x}.", "Can you explain {x}?", "Could you explain {x} to me?",
            "What can you tell me about {x}?", "Give me information about {x}.", "Give me an overview of {x}.",
            "Describe {x}.", "I want to know about {x}.", "I'd like to know about {x}", "Help me understand {x}.",
            "Can you give me a simple explanation of {x}?", "Could you tell me about {x}?",
            "I want some information about {x}.", "Please tell me about {x}", "What do you know about {x}?"]


def _subject(topic):
    return re.sub(r"(?i)^(a|an|the)\s+", "", topic)


@pytest.mark.parametrize("topic", TOPICS)
def test_every_wording_of_a_request_resolves_to_the_same_topic(topic):
    subject = _subject(topic)
    for w in WORDINGS:
        u = C.understand(w.format(x=topic), [])
        assert u.kind == "knowledge", (w, u)
        assert u.topics == [subject], (w, u.topics)
        assert re.sub(r"(?i)\b(a|an|the)\s+", "", u.resolved).lower().rstrip("?") in (
            f"what is {subject}".lower(), f"what are {subject}".lower()), (w, u.resolved)


@pytest.mark.parametrize("topic", TOPICS)
def test_the_wording_does_not_change_which_question_is_asked(topic):
    resolved = {C.understand(w.format(x=topic), []).resolved for w in WORDINGS}
    assert len(resolved) == 1, resolved


def test_a_request_that_already_names_an_aspect_keeps_it():
    u = C.understand("Tell me about the inputs and outputs of a system", [])
    assert "inputs" in u.resolved and u.topics == ["system"]
    u = C.understand("Explain how photosynthesis works", [])
    assert u.resolved == "How does photosynthesis work?"
    u = C.understand("Explain what a CPU does.", [])
    assert u.resolved == "What does CPU do?"


# ---- C. follow-ups --------------------------------------------------------------------------------------------------
def test_its_refers_to_the_current_topic():
    u = run(["What is a system?", "What are its inputs and outputs?"])[1]
    assert u.kind == "followup" and u.resolved == "What are system's inputs and outputs?"


@pytest.mark.parametrize("subject,asks", [
    ("DNA", ["What does it do?", "Where is it found?", "Why is it important?", "What is it made of?"]),
    ("photosynthesis", ["Why is it important?", "Where does it happen?", "Why is this important?"]),
    ("a CPU", ["What does it do?", "What are its main parts?", "Give me an example."]),
    ("HTTP", ["How does it work?", "Who created it?"]),
])
def test_follow_ups_resolve_to_the_subject(subject, asks):
    plain = _subject(subject)
    for u in run([f"What is {subject}?"] + asks)[1:]:
        assert u.kind == "followup" and plain in u.resolved, u


def test_the_computer_chain():
    chain = run(["Hi", "Tell me about computers.", "What are their inputs?", "What are their outputs?",
                 "How does it process them?", "Give me an example.", "Thanks"])
    kinds = [u.kind for u in chain]
    assert kinds == ["casual", "knowledge", "followup", "followup", "followup", "followup", "casual"]
    assert chain[1].resolved == "What are computers?"
    assert all("computers" in u.resolved for u in chain[2:6])
    assert "inputs" in chain[2].resolved and "outputs" in chain[3].resolved


def test_ellipsis_with_no_pronoun_still_uses_the_topic():
    u = run(["What is gravity?", "Give me an example."])[1]
    assert u.kind == "followup" and u.resolved == "Give me an example of gravity"
    u = run(["What is gravity?", "Why?"])[1]
    assert u.kind == "followup" and "gravity" in u.resolved


def test_a_demonstrative_with_a_noun_names_its_own_subject():
    u = run(["What is a system?", "What are the parts of this system?"])[1]
    assert u.kind == "knowledge" and u.topics == ["system"]       # it names its own subject: not a follow-up


# ---- D. topic switching ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("first,second,expect", [("DNA", "a CPU", "CPU"), ("a CPU", "DNA", "DNA"),
                                                 ("photosynthesis", "HTTP", "HTTP"), ("gravity", "a database", "database")])
def test_the_pronoun_follows_the_most_recent_subject(first, second, expect):
    u = run([f"What is {first}?", f"What is {second}?", "What does it do?"])[2]
    assert u.kind == "followup" and expect in u.resolved and _subject(first) not in u.resolved


def test_a_standalone_question_never_inherits_the_old_topic():
    chain = run(["What is DNA?", "What is the capital of Japan?", "Who wrote Hamlet?"])
    assert chain[1].kind == "knowledge" and "DNA" not in chain[1].resolved
    assert chain[2].kind == "knowledge" and "DNA" not in chain[2].resolved and "Japan" not in chain[2].resolved
    # and the reference after a standalone question points at THAT question's subject
    assert "Japan" in run(["What is DNA?", "What is the capital of Japan?", "What is its population?"])[2].resolved


# ---- E. conversations are isolated (history is the only input) -------------------------------------------------------
def test_a_conversation_without_history_cannot_borrow_one():
    a = run(["What is DNA?"])
    assert C.understand("What does it do?", []).kind == "clarify"          # a fresh conversation: nothing to refer to
    assert C.understand("What does it do?", [{"topics": a[0].topics}]).kind == "followup"


# ---- clarification instead of guessing ------------------------------------------------------------------------------
def test_a_reference_with_two_candidates_asks_which():
    chain = run(["How is a CPU different from a GPU?", "What does it do?"])
    assert chain[1].kind == "clarify" and chain[1].reply == "Do you mean the CPU or the GPU?"
    assert run(["How is a CPU different from a GPU?", "What do they do?"])[1].kind == "followup"


def test_a_reference_with_nothing_to_refer_to_asks():
    u = C.understand("Where is it found?", [])
    assert u.kind == "clarify" and "referring" in u.reply


# ---- no hard-coding --------------------------------------------------------------------------------------------------
def test_the_module_names_no_subject():
    src = pathlib.Path(C.__file__).read_text(encoding="utf-8").lower()
    src = re.sub(r"_reply = \{.*?\n\}\n", "", src, flags=re.S)   # replies shown to people may give examples
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    for subject in ("dna", "cpu", "photosynthesis", "computer", "http", "gravity", "database", "japan", "hamlet"):
        assert not re.search(rf"(?<![a-z]){subject}(?![a-z])", code), subject
