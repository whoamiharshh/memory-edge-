"""Source separation and no fake evidence.

Every answer names where its evidence came from, in four separate words: device, fleet, offline_kb, web. The
local model is never one of them. When nothing is found, the model may still write an answer, but that answer
is ungrounded, cites nothing and says so; it must never be passed off as coming from any of the four.
"""
import socket

import httpx
import pytest

from edge import online, seed


class Quote:
    """A model that would happily invent a citation if the code let it."""
    available = True

    def __init__(self, reply="It is probably Lisbon. [E1]", raises=False):
        self.reply, self.raises, self.chat_calls = reply, raises, 0

    def chat(self, q, max_tokens=200):
        self.chat_calls += 1
        if self.raises:
            raise RuntimeError("model failed")
        return self.reply

    def complete(self, prompt, max_tokens=180):
        if self.raises:
            raise RuntimeError("model failed")
        return self.reply


@pytest.fixture
def dev(make_device):
    d, _ = make_device("solo", "site1")
    d.outbox.kv_set("online", False)
    return d


def _shape(out):
    return out["mode"], out["grounded"], out["sources"], [u["origin"] for u in out["used"]]


def test_device_memory_is_labelled_device(dev):
    dev.remember("The spare bearing kit is kept in cabinet 7.", kind="fact")
    out = dev.ask("where is the spare bearing kit kept", None)
    assert out["grounded"] is True and out["sources"] == ["device"]
    assert {u["origin"] for u in out["used"]} == {"device"}


def test_fleet_evidence_is_labelled_fleet(dev, monkeypatch):
    case = {"component": "bearing", "fault_class": "inner_race", "n_sites": 2, "n_events": 3, "flags": [], "notes": [],
            "actions": [{"action_code": "replace_bearing", "sites_worked": ["a", "b"], "sites_failed": [],
                         "machine_verified": 2}]}
    monkeypatch.setattr(dev, "search", lambda **kw: {"local": [], "fleet": [{"id": "f1", "case": case}],
                                                     "latency_ms": 0})
    out = dev.ask("bearing inner race replace", None)
    assert out["grounded"] is True and out["sources"] == ["fleet"]


def test_offline_library_is_labelled_offline_kb_with_its_citation(dev):
    seed.seed_if_empty(dev, ["general_facts"])
    out = dev.ask("What is the capital of Japan?", Quote())
    assert out["grounded"] is True and out["sources"] == ["offline_kb"]
    ref = out["used"][0]
    assert ref["origin"] == "offline_kb" and ref["sources"][0].startswith("https://www.wikidata.org/")


def test_web_evidence_is_labelled_web_and_links_its_page(make_device, monkeypatch):
    monkeypatch.setenv("EDGE_SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setattr(online, "search", lambda q, limit=5: [
        {"title": "Quokka", "url": "https://example.org/quokka", "snippet": "The quokka is a small marsupial."}])
    d, _ = make_device("solo", "site1")
    out = d.ask("what is a quokka", None)
    assert out["grounded"] is True and out["sources"] == ["web"]
    assert out["used"][0]["url"] == "https://example.org/quokka"


def test_nothing_found_is_never_presented_as_evidence(dev):
    """The model would invent a citation; it is never even asked, so nothing becomes a source or grounded."""
    seed.seed_if_empty(dev, ["general_facts"])
    q = Quote()
    out = dev.ask("Who painted the ceiling of the zxqv chapel?", q)
    assert _shape(out) == ("no_evidence", False, [], [])
    assert q.chat_calls == 0 and "internet connection" in out["answer"]


def test_a_model_sentence_that_adds_facts_is_dropped_and_the_evidence_quoted(dev):
    dev.remember("The spare bearing kit is kept in cabinet 7.", kind="fact")
    inventive = Quote("The spare bearing kit is kept near the loading dock lift in cabinet 7 [E1].")
    out = dev.ask("where is the spare bearing kit kept", inventive)
    assert out["mode"] == "quoted" and "loading dock" not in out["answer"]
    assert "cabinet 7" in out["answer"] and out["grounded"] is True


def test_a_faithful_model_sentence_is_kept(dev):
    dev.remember("The spare bearing kit is kept in cabinet 7.", kind="fact")
    out = dev.ask("where is the spare bearing kit kept", Quote("The spare bearing kit is kept in cabinet 7 [E1]."))
    assert out["mode"] == "llm" and out["answer"] == "The spare bearing kit is kept in cabinet 7 [E1]."


def test_a_number_ending_a_sentence_in_the_evidence_is_recognised():
    """Regression: 'cabinet 7.' in the evidence was read as a decimal, so a correct '7' was rejected."""
    from edge import rag
    ev = {"E1": "The spare bearing kit is kept in cabinet 7."}
    assert rag.check_output("The kit is in cabinet 7 [E1].", {"E1"}, ev)[0]
    assert not rag.check_output("The kit is in cabinet 8 [E1].", {"E1"}, ev)[0]       # a wrong number still fails


def test_library_text_is_quoted_never_paraphrased(dev):
    seed.seed_if_empty(dev, ["general_facts"])
    out = dev.ask("What is the capital of Japan?", Quote("Tokyo is the capital of Japan, a big city. [E1]"))
    assert out["mode"] == "quoted" and "The capital of Japan is Tokyo." in out["answer"]


def test_a_library_entry_covering_little_of_the_question_is_not_evidence(dev, monkeypatch, tmp_path):
    """Regression: with a large library some article shares a word with every question. 'Who painted the ceiling
    of the zxqv chapel?' was answered as grounded from articles that only mention 'chapel' or 'painted'."""
    import json
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "simplewiki_leads.jsonl").write_text("\n".join(json.dumps(a) for a in [
        {"title": "Sistine Chapel", "text": "The Sistine Chapel is a chapel in Vatican City, the home of the Pope.",
         "url": "https://simple.wikipedia.org/wiki/Sistine_Chapel"},
        {"title": "Mona Lisa", "text": "The Mona Lisa is a portrait painting by the Italian artist Leonardo.",
         "url": "https://simple.wikipedia.org/wiki/Mona_Lisa"}]), encoding="utf-8")
    monkeypatch.setattr(seed, "KNOWLEDGE", tmp_path / "knowledge")
    seed.seed_if_empty(dev, ["simple_wikipedia"])

    thin = dev.ask("Who painted the ceiling of the zxqv chapel?", Quote("I do not know."))
    assert thin["grounded"] is False and thin["used"] == [] and thin["sources"] == []
    covered = dev.ask("What is the Sistine Chapel?", None)
    assert covered["grounded"] is True and covered["sources"] == ["offline_kb"]


def test_nothing_found_and_model_fails_is_the_honest_message(dev):
    out = dev.ask("Who painted the ceiling of the zxqv chapel?", Quote(raises=True))
    assert _shape(out) == ("no_evidence", False, [], []) and out["answer"].startswith("Needs internet connection for this.")


def test_model_failure_with_evidence_still_cites_the_evidence(dev):
    seed.seed_if_empty(dev, ["general_facts"])
    out = dev.ask("What is the capital of Japan?", Quote(raises=True))
    assert out["grounded"] is True and out["mode"] == "quoted" and "Tokyo" in out["answer"]
    assert out["sources"] == ["offline_kb"]


def test_qdrant_failure_degrades_instead_of_crashing(dev, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("qdrant is down")
    monkeypatch.setattr(dev.store, "search", boom)
    out = dev.ask("What is the capital of Japan?", Quote("Tokyo."))
    assert out["grounded"] is False and out["used"] == [] and out["sources"] == []
    assert out["memory_failed"], "the answer must say a search leg was lost"
    assert out["answer"].startswith("Searching")


def test_qdrant_failure_without_a_model_is_still_an_answer(dev, monkeypatch):
    monkeypatch.setattr(dev.store, "search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    out = dev.ask("What is the capital of Japan?", None)
    assert out["mode"] == "no_evidence" and out["memory_failed"]


def test_switch_says_online_but_the_network_is_dead_reports_offline(make_device, monkeypatch):
    monkeypatch.setenv("EDGE_SEARCH_PROVIDER", "duckduckgo")

    def dead(*a, **k):
        raise httpx.ConnectError("no network")
    monkeypatch.setattr(httpx.Client, "post", dead)
    monkeypatch.setattr(httpx.Client, "get", dead)
    d, _ = make_device("solo", "site1")                        # the online switch is ON
    out = d.ask("what is a quokka", None)
    assert out["web_reason"] == "offline" and out["needs_internet"] is True
    assert out["answer"].startswith("Needs internet connection for this.")


@pytest.mark.parametrize("switch,mode", [(False, "auto"), (True, "local")])
def test_offline_or_local_only_never_opens_a_network_connection(make_device, monkeypatch, switch, mode):
    monkeypatch.setenv("EDGE_SEARCH_PROVIDER", "duckduckgo")
    attempts = []

    def spy(self, addr):
        attempts.append(addr)
        raise OSError("blocked by test")
    d, _ = make_device("solo", "site1")
    d.outbox.kv_set("online", switch)
    d.set_retrieval_mode(mode)
    monkeypatch.setattr(socket.socket, "connect", spy)
    monkeypatch.setattr(socket.socket, "connect_ex", lambda self, addr: attempts.append(addr) or 1)
    d.ask("what is a quokka", Quote("A marsupial."))
    d.ask("What problems have you seen?", None)
    assert attempts == [], f"a question left the device: {attempts}"
