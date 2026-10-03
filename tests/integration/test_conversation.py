"""Whole conversations through Device.ask: casual talk, paraphrases, follow-ups, topic switches, isolation, sources.

The library rows below are TEST DATA (short passages written like encyclopedia leads); the product code contains none of them.
The assertions look at what the device actually did - which message type it detected, what question it resolved to, what it
retrieved and where the answer came from - through the trace every answer carries.
"""
import pytest

from edge import seed

DNA = ("wiki:DNA", "DNA", "DNA: DNA, short for deoxyribonucleic acid, is the molecule that contains the genetic code of "
       "living organisms. It is found in the nucleus of cells.", "Simple English Wikipedia", [])
CPU = ("tech:Central processing unit", "Central processing unit",
       "Central processing unit: A central processing unit (CPU) is the part of a computer that carries out the "
       "instructions of a program. It performs arithmetic, logic, controlling and input/output operations.",
       "Technical concepts (Wikipedia)", ["CPU", "Central processing unit"])
COMPUTER = ("wiki:Computer", "Computer", "Computer: A computer is a machine that can be programmed to carry out sequences "
            "of arithmetic or logical operations. Computers take input, process it and produce output.",
            "Simple English Wikipedia", [])
PHOTO = ("wiki:Photosynthesis", "Photosynthesis", "Photosynthesis: Photosynthesis is the process by which green plants make "
         "food from sunlight. It happens in the chloroplasts of leaves.", "Simple English Wikipedia", [])
JAPAN = ("capital:Q17", "Capital of Japan", "The capital of Japan is Tokyo.", "general knowledge", [])


class Forbidden:
    """A model that must not be asked anything."""
    available = True

    def __getattr__(self, name):
        raise AssertionError(f"the model must not be used here: .{name}")


@pytest.fixture
def dev(make_device):
    d, _ = make_device("solo", "site1")
    d.outbox.kv_set("online", False)
    seed._flush(d, [{"ref_id": r[0], "title": r[1], "text": r[2], "sources": ["https://example.org/" + r[0]],
                     "topic": r[3], "pack": "test", "aliases": r[4]} for r in (DNA, CPU, COMPUTER, PHOTO, JAPAN)])
    return d


def chat(dev, session, *messages, llm=None):
    return [dev.ask(m, llm, session=session) for m in messages]


# ---- A. casual conversation never reaches retrieval ----------------------------------------------------------------------
@pytest.mark.parametrize("msg", ["Hi", "Hello", "Hey", "Good morning", "How are you?", "Thanks", "Thank you", "Bye"])
def test_casual_messages_are_answered_without_retrieval_or_a_model(dev, msg):
    out = dev.ask(msg, Forbidden(), session="s-casual")
    assert out["mode"] == "smalltalk" and out["used"] == [] and out["sources"] == []
    assert out["grounded"] is True and "internet" not in out["answer"].lower() and "evidence" not in out["answer"].lower()
    assert out["trace"]["message_type"] == "casual"
    assert out["retrieval"] is None


def test_hi_then_a_question_then_thanks(dev):
    out = chat(dev, "s-1", "Hi", "Tell me about computers.", "What are their inputs?", "Thanks")
    assert [o["trace"]["message_type"] for o in out] == ["casual", "knowledge", "followup", "casual"]
    assert out[1]["trace"]["resolved"] == "What are computers?" and out[1]["sources"] == ["offline_kb"]
    assert "computers" in out[2]["trace"]["resolved"] and "inputs" in out[2]["trace"]["resolved"]


# ---- B/F. the same request in different words retrieves the same thing --------------------------------------------------
@pytest.mark.parametrize("wordings,title", [
    (["What is DNA?", "Tell me about DNA.", "Explain DNA.", "Can you explain DNA?", "What can you tell me about DNA?",
      "Give me an overview of DNA.", "I want to know about DNA.", "Help me understand DNA.", "Describe DNA."], "DNA"),
    (["What is photosynthesis?", "Tell me about photosynthesis.", "Explain photosynthesis.", "Could you explain photosynthesis?",
      "Give me information about photosynthesis."], "Photosynthesis"),
    (["What is a CPU?", "Tell me about CPUs.", "Explain what a CPU does.", "Can you tell me about a CPU?",
      "I'd like to know about the CPU."], "Central processing unit"),
])
def test_different_wordings_give_the_same_answer_from_the_same_source(dev, wordings, title):
    seen = []
    for i, w in enumerate(wordings):
        out = dev.ask(w, None, session=f"s-paraphrase-{i}")
        assert out["grounded"] is True and out["sources"] == ["offline_kb"], (w, out["answer"])
        assert [u["title"] for u in out["used"]] == [title], (w, out["used"])
        seen.append(out["answer"])
    assert len(set(seen)) <= 2           # the same lead (or the same lead with and without its answering sentence)


# ---- C. follow-ups ------------------------------------------------------------------------------------------------------
def test_follow_ups_are_resolved_before_retrieval(dev):
    out = chat(dev, "s-dna", "What is DNA?", "What does it do?", "Where is it found?", "Why is it important?")
    resolved = [o["trace"]["resolved"] for o in out]
    assert resolved[1] == "What does DNA do?" and resolved[2] == "Where is DNA found?" and resolved[3] == "Why is DNA important?"
    assert [o["trace"]["message_type"] for o in out] == ["knowledge", "followup", "followup", "followup"]
    # and the answer to "where is it found" really comes from the DNA passage
    assert out[2]["used"] and out[2]["used"][0]["title"] == "DNA" and "nucleus" in out[2]["answer"]


def test_its_inputs_and_outputs(dev):
    out = chat(dev, "s-sys", "What is a computer?", "What are its inputs and outputs?", "Give me an example.")
    assert out[1]["trace"]["resolved"] == "What are computer's inputs and outputs?"
    assert out[2]["trace"]["resolved"] == "Give me an example of computer"


# ---- D. topic switching ----------------------------------------------------------------------------------------------
def test_it_means_the_most_recent_subject(dev):
    a = chat(dev, "s-a", "What is DNA?", "What is a CPU?", "What does it do?")
    assert a[2]["trace"]["resolved"] == "What does CPU do?" and a[2]["used"][0]["title"] == "Central processing unit"
    b = chat(dev, "s-b", "What is a CPU?", "What is DNA?", "Where is it found?")
    assert b[2]["trace"]["resolved"] == "Where is DNA found?" and b[2]["used"][0]["title"] == "DNA"


def test_a_standalone_question_does_not_inherit_the_old_topic(dev):
    out = chat(dev, "s-standalone", "What is DNA?", "What is the capital of Japan?")
    assert out[1]["trace"]["message_type"] == "knowledge" and "DNA" not in out[1]["trace"]["resolved"]
    assert out[1]["answer"].startswith("The capital of Japan is Tokyo.")


# ---- E. conversation isolation, by real session ids -------------------------------------------------------------------
def test_two_conversations_never_share_context(dev):
    chat(dev, "conv-A", "What is DNA?")
    out = chat(dev, "conv-B", "What is a CPU?", "What does it do?")
    assert out[1]["trace"]["resolved"] == "What does CPU do?"
    assert out[1]["session"] == "conv-B" and out[0]["session"] == "conv-B"
    # a fresh conversation has nothing to refer to, even though another one is about DNA
    fresh = dev.ask("What does it do?", None, session="conv-C")
    assert fresh["mode"] == "clarify" and "referring" in fresh["answer"]
    assert fresh["used"] == [] and fresh["grounded"] is False


def test_turns_are_stored_under_the_session_that_was_sent(dev):
    chat(dev, "conv-X", "What is DNA?")
    chat(dev, "conv-Y", "What is a CPU?")
    x = [t["note_text"] for t in dev.recent_chat(10, session="conv-X")]
    y = [t["note_text"] for t in dev.recent_chat(10, session="conv-Y")]
    assert len(x) == 1 and "DNA" in x[0] and len(y) == 1 and "CPU" in y[0]
    assert dev.recent_chat(10, session="conv-X")[0]["topics"] == ["DNA"]


def test_without_a_session_the_devices_current_one_is_used_and_returned(dev):
    first = dev.ask("What is DNA?", None)
    second = dev.ask("What does it do?", None)
    assert first["session"] == second["session"] == dev.current_session()
    assert second["trace"]["message_type"] == "followup"
    dev.forget_chat()                                        # "New conversation"
    third = dev.ask("What does it do?", None)
    assert third["session"] != first["session"] and third["mode"] == "clarify"


def test_a_malformed_session_id_is_ignored_not_trusted(dev):
    out = dev.ask("What is DNA?", None, session="../../etc/passwd")
    assert out["session"] == dev.current_session()


# ---- ambiguity is asked, never guessed ------------------------------------------------------------------------------------
def test_an_ambiguous_reference_gets_a_question(dev):
    out = chat(dev, "s-amb", "How is a CPU different from a computer?", "What does it do?")
    assert out[1]["mode"] == "clarify" and out[1]["answer"] == "Do you mean the CPU or the computer?"
    # answering the clarification works as a normal question
    assert chat(dev, "s-amb", "the CPU")[0]["trace"]["message_type"] == "knowledge"


# ---- sources stay honest -------------------------------------------------------------------------------------------------
def test_the_library_is_never_called_what_the_device_holds(dev):
    for q in ("Tell me about DNA.", "What is the capital of Japan?", "Explain photosynthesis."):
        out = dev.ask(q, None, session="s-src")
        assert out["sources"] == ["offline_kb"] and "this device holds" not in out["answer"]
        assert {u["origin"] for u in out["used"]} == {"offline_kb"}


def test_device_memory_is_still_labelled_device_memory(dev):
    dev.remember("The spare coupling for pump 7 is in cabinet 3.", kind="fact")
    out = dev.ask("Where is the spare coupling for pump 7?", None, session="s-dev")
    assert out["sources"] == ["device"] and out["answer"].startswith("From what this device holds")


def test_a_topic_the_library_lacks_is_declined_honestly_not_as_a_misread(dev):
    out = dev.ask("Tell me about the Ottoman fleet in 1571.", None, session="s-none")
    assert out["answer"] == "Needs internet connection for this." and out["used"] == []
    assert out["trace"]["message_type"] == "knowledge"        # understood fine; the device simply does not hold it


# ---- the trace tells you what happened --------------------------------------------------------------------------------------
def test_every_answer_carries_a_trace(dev):
    out = dev.ask("Tell me about DNA.", None, session="s-trace")
    t = out["trace"]
    for key in ("session", "message_type", "original", "resolved", "topics", "context_turns", "retrieval_query", "mode",
                "sources", "candidates", "selected"):
        assert key in t, key
    assert t["original"] == "Tell me about DNA." and t["resolved"] == "What is DNA?" and t["topics"] == ["DNA"]
    assert t["selected"][0]["title"] == "DNA"


def test_trace_logging_can_never_break_an_answer(dev, monkeypatch, capsys):
    """A non-ASCII character in a passage crashed the console logger on Windows and turned a good answer into an error."""
    monkeypatch.setenv("EDGE_ASK_TRACE", "1")
    seed._flush(dev, [{"ref_id": "wiki:Dvořák", "title": "Dvořák", "text": "Dvořák: Antonín Dvořák was a Czech composer.",
                       "sources": ["https://example.org/d"], "topic": "Simple English Wikipedia", "pack": "test",
                       "aliases": []}])
    out = dev.ask("Who was Dvořák?", None, session="s-log")
    assert out["answer"] and out["sources"] == ["offline_kb"]
    assert "[ask-trace]" in capsys.readouterr().out
