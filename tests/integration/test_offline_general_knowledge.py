"""Offline with nothing relevant in memory: the device says it needs a connection. It never lets the local model
answer from its own training, because a small model is sometimes confidently wrong.

The model is a stub that fails the test if it is asked anything.
"""
from edge import rag


class ForbiddenLLM:
    available = True

    def __getattr__(self, name):
        raise AssertionError(f"the model must not be used when nothing matched: .{name}")


def _offline(make_device, name="solo"):
    d, _ = make_device(name, "site1")
    d.outbox.kv_set("online", False)
    return d


def test_no_evidence_offline_is_a_needs_internet_message_not_a_model_answer(make_device):
    out = _offline(make_device).ask("What is the capital of Japan?", ForbiddenLLM())
    assert (out["mode"], out["grounded"], out["sources"], out["used"]) == ("no_evidence", False, [], [])
    assert out["needs_internet"] is True and "internet connection" in out["answer"]
    assert out["model"] is None


def test_relevant_memory_still_wins_and_stays_grounded(make_device):
    d = _offline(make_device)
    d.remember("The spare bearing kit is kept in cabinet 7.", kind="fact")
    out = d.ask("where is the spare bearing kit kept", None)
    assert out["grounded"] is True and out["used"] and out["mode"] == "quoted"


def test_no_model_gives_the_same_honest_answer(make_device):
    out = _offline(make_device).ask("What is the capital of Japan?", None)
    assert (out["mode"], out["grounded"], out["sources"], out["used"]) == ("no_evidence", False, [], [])


def test_unavailable_model_file_is_harmless(make_device, tmp_path):
    out = _offline(make_device).ask("What is the capital of Japan?", rag.LocalLLM(path=tmp_path / "nope.gguf"))
    assert out["mode"] == "no_evidence"
