"""Free-text memory and the conversational `ask`.

The behaviour under test is the one that makes the device honest: retrieval always returns *something*,
so a question that matches nothing must still be answered with "I do not know" rather than with the
nearest record. Everything here runs without an LLM, so `ask` falls back to quoting the evidence.
"""
import pytest


def test_remember_and_recall_arbitrary_text(make_device):
    d, _ = make_device("devA", "site1")
    d.remember("The spare keys are in the second drawer of the blue cabinet.")
    d.remember("Deliveries arrive on Tuesday and Friday mornings.")

    hits = d.recall("where are the keys kept", limit=5)
    assert hits, "a plain-language question should find the fact that answers it"
    assert any("second drawer" in h["note_text"] for h in hits)


def test_memory_is_local_only(make_device):
    """A memory must never be queued for the cloud: it has had no outcome verification."""
    d, _ = make_device("devA", "site1")
    before = d.outbox.counts()
    d.remember("Something private about this site.")
    assert d.outbox.counts() == before
    assert d.list_memories("fact")[0]["share_state"] == "local"


def test_unknown_question_is_answered_with_i_do_not_know(make_device):
    d, _ = make_device("devA", "site1")
    d.remember("The spare keys are in the second drawer of the blue cabinet.")

    out = d.ask("what is the capital of Peru")
    assert out["grounded"] is False
    assert out["used"] == []
    assert out["answer"].startswith("Needs internet connection for this.")


def test_it_distinguishes_a_near_miss_from_something_it_could_never_know(make_device):
    """Two different failures that look identical from outside. "Nothing matched" asks the person to
    rephrase; "this is outside what I hold" tells them it needs teaching or a connection, because no
    amount of rephrasing will ever find it."""
    d, _ = make_device("devA", "site1")
    d.remember("Plot 44 is 2.3 acres and the transfer completed in March.")

    far = d.ask("who won the world cup in 1998")
    assert far["needs_internet"] is True
    assert far["answer"].startswith("Needs internet connection for this.")
    # and it says WHICH reason it could not look it up, rather than implying it never could. Under test
    # the provider is forced off (conftest), so that is the reason reported here.
    assert far["web_reason"] == "no web search configured"
    assert "no web search is configured" in far["answer"].lower()

    near = d.ask("how big is plot 91")           # shares "plot" with a stored record, but not that plot
    assert near["answer"].startswith("Needs internet connection for this.")


def test_known_question_is_grounded_in_the_stored_fact(make_device):
    d, _ = make_device("devA", "site1")
    d.remember("The spare keys are in the second drawer of the blue cabinet.")

    out = d.ask("where are the spare keys")
    assert out["grounded"] is True
    assert out["used"] and out["used"][0]["source"] == "memory"
    assert "[E1]" in out["answer"]


def test_a_specific_number_must_match_or_the_answer_is_refused(make_device):
    """Word overlap alone would answer "plot 91" with the record for "plot 44": every ordinary word
    matches. An identifier in the question has to be present in the record."""
    d, _ = make_device("devA", "site1")
    d.remember("Plot 44 is 2.3 acres and the transfer completed in March.")

    assert d.ask("how big is plot 44")["grounded"] is True
    assert d.ask("how big is plot 91")["grounded"] is False


def test_each_turn_is_stored_and_replayable(make_device):
    d, _ = make_device("devA", "site1")
    d.ask("first question")
    d.ask("second question")

    turns = d.recent_chat(10)
    assert len(turns) == 2
    assert turns[0]["note_text"].startswith("Q: first question")
    assert turns[-1]["note_text"].startswith("Q: second question")


def test_past_turns_are_not_quoted_back_as_evidence(make_device):
    """A turn is context for follow-ups, never a source. Without this the device cites its own earlier
    answer and launders a guess into a fact."""
    d, _ = make_device("devA", "site1")
    d.ask("tell me about widgets")          # stored as a chat turn containing the word "widgets"

    out = d.ask("tell me about widgets")
    assert out["grounded"] is False, "its own previous turn must not count as evidence"


def test_clearing_the_conversation_keeps_what_was_taught(make_device):
    d, _ = make_device("devA", "site1")
    d.remember("Deliveries arrive on Tuesday.")
    d.ask("when do deliveries arrive")
    assert d.recent_chat(10)

    assert d.forget_chat() == 1
    assert d.recent_chat(10) == []
    assert len(d.list_memories("fact")) == 1


def test_memory_survives_a_restart(make_device):
    d, _ = make_device("devA", "site1")
    d.remember("The gate code is written inside the cupboard door.")
    d.close()

    d2, _ = make_device("devA", "site1")
    assert any("gate code" in m["note_text"] for m in d2.list_memories("fact"))


@pytest.mark.parametrize("bad", ["", "   ", "x" * 5000])
def test_bad_memory_text_is_refused(make_device, bad):
    d, _ = make_device("devA", "site1")
    with pytest.raises(ValueError):
        d.remember(bad)


# ---- follow-up questions -------------------------------------------------------------------------------

def test_a_follow_up_inherits_the_subject_of_the_conversation(make_device):
    """"how does it work" carries no subject of its own. Without the conversation it finds nothing."""
    d, _ = make_device("devA", "site1")
    d.remember("The irrigation pump runs on a timer and starts at 5am every day.")

    first = d.ask("tell me about the irrigation pump")
    assert first["grounded"] is True

    follow = d.ask("when does it start")
    assert follow["grounded"] is True, "a follow-up should resolve against the previous question"
    assert "5am" in follow["answer"] or "5am" in follow["used"][0]["text"]


def test_a_self_contained_question_does_not_inherit_the_previous_subject(make_device):
    """The mirror image, and the more dangerous direction: a complete question must keep to its own
    subject, or every later question quietly drags the earlier topic along."""
    d, _ = make_device("devA", "site1")
    d.remember("The irrigation pump runs on a timer and starts at 5am every day.")

    d.ask("tell me about the irrigation pump")
    out = d.ask("what is the capital of Peru")
    assert out["grounded"] is False


def test_a_question_naming_its_own_identifier_never_inherits_another(make_device):
    """Regression: "how" was briefly treated as a referring word, so "how big is plot 91" inherited
    "plot 44" from the turn before and answered about the wrong plot."""
    d, _ = make_device("devA", "site1")
    d.remember("Plot 44 is 2.3 acres and the transfer completed in March.")

    assert d.ask("how big is plot 44")["grounded"] is True
    assert d.ask("how big is plot 91")["grounded"] is False


def test_follow_up_still_refuses_when_the_conversation_leads_nowhere(make_device):
    d, _ = make_device("devA", "site1")
    d.ask("tell me about something never stored")
    assert d.ask("how does it work")["grounded"] is False


def test_a_one_word_subject_is_a_new_question_not_a_follow_up(make_device):
    """Regression: "what is harsh" has a single content word, was treated as a follow-up, inherited the
    previous topic and answered a question about a person with a vehicle fault code."""
    d, _ = make_device("devA", "site1")
    d.remember("P0420 means the catalytic converter efficiency is below threshold on bank 1.")

    assert d.ask("what does P0420 mean")["grounded"] is True
    assert d.ask("what is harsh")["grounded"] is False


# ---- conversations -------------------------------------------------------------------------------------

def test_clearing_starts_a_new_conversation_and_keeps_the_old_one(make_device):
    """On a chat screen "clear" means "give me a blank page", not "destroy what I said" — and the history
    is what makes follow-up questions work, so it is kept and grouped."""
    d, _ = make_device("devA", "site1")
    d.remember("The generator is behind the workshop.")
    d.ask("where is the generator")
    d.forget_chat()
    d.ask("what is stored here")

    convs = d.conversations()
    assert len(convs) == 2, "each conversation is listed separately"
    assert convs[0]["title"] == "what is stored here"
    assert sum(c["turns"] for c in convs) == 2


def test_a_past_conversation_can_be_reopened(make_device):
    d, _ = make_device("devA", "site1")
    d.ask("first question")
    d.forget_chat()
    d.ask("second question")

    oldest = d.conversations()[-1]
    turns = d.conversation(oldest["session"])
    assert len(turns) == 1 and turns[0]["note_text"].startswith("Q: first question")


def test_a_new_conversation_does_not_inherit_the_old_subject(make_device):
    """Follow-up context must not leak across the boundary, or clearing the chat would achieve nothing."""
    d, _ = make_device("devA", "site1")
    d.remember("The irrigation pump starts at 5am.")
    d.ask("tell me about the irrigation pump")
    d.forget_chat()

    assert d.ask("when does it start")["grounded"] is False


def test_deleting_really_erases_the_turns(make_device):
    d, _ = make_device("devA", "site1")
    d.ask("something")
    assert d.forget_chat(delete=True) == 1
    assert d.conversations() == []


def test_the_answer_reports_what_qdrant_searched(make_device):
    """The Qdrant screen shows this verbatim, so it has to describe the search that actually ran."""
    d, _ = make_device("devA", "site1")
    d.remember("The spare keys are in the blue cabinet.")

    r = d.ask("where are the keys")["retrieval"]
    assert r["total_points"] >= 1
    assert {s["source"] for s in r["searched"]} >= {"memory", "shared", "reference", "record"}
    assert any(v["name"] == "note_bm25" and v["used"] for v in r["vectors"])
    assert "memory" in r["kept_sources"]
