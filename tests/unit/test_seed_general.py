"""The general-knowledge reference packs: loaded into memory, found by Ask, cited, and never grounded on a guess."""
import json

from edge import seed


class NeverCalledLLM:
    available = True

    def chat(self, question, max_tokens=200):
        raise AssertionError("the model must not answer from its own training")

    def complete(self, prompt, max_tokens=180):
        raise AssertionError("library text is quoted verbatim, never paraphrased by the model")


def test_capital_is_answered_from_the_pack_with_a_citation(make_device):
    d, _ = make_device("solo", "site1")
    d.outbox.kv_set("online", False)
    seed.seed_if_empty(d, ["general_facts"])
    out = d.ask("What is the capital of Japan?", NeverCalledLLM())
    assert out["grounded"] is True and out["mode"] == "quoted"      # the library is quoted word for word
    assert "Tokyo" in out["answer"]
    ref = [u for u in out["used"] if u["source"] == "reference"][0]
    assert ref["sources"] and ref["sources"][0].startswith("https://www.wikidata.org/")


def test_missing_wikipedia_file_is_retried_not_marked_done(make_device, monkeypatch, tmp_path):
    d, _ = make_device("solo", "site1")
    monkeypatch.setattr(seed, "KNOWLEDGE", tmp_path / "knowledge")         # the leads file then lives in tmp_path/data/raw
    assert seed.seed_if_empty(d, ["simple_wikipedia"])["seeded"] == 0
    assert "simple_wikipedia" not in seed.seeded(d)

    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "simplewiki_leads.jsonl").write_text(json.dumps(
        {"title": "Bearing", "text": "A bearing is a machine element that constrains relative motion.",
         "url": "https://simple.wikipedia.org/wiki/Bearing"}) + "\n", encoding="utf-8")
    assert seed.seed_if_empty(d, ["simple_wikipedia"])["seeded"] == 1
    assert "simple_wikipedia" in seed.seeded(d)


def test_unanswerable_question_is_not_answered_by_the_model(make_device):
    d, _ = make_device("solo", "site1")
    d.outbox.kv_set("online", False)
    seed.seed_if_empty(d, ["general_facts"])
    out = d.ask("Who painted the ceiling of the zxqv chapel?", NeverCalledLLM())
    assert (out["mode"], out["grounded"], out["used"], out["sources"]) == ("no_evidence", False, [], [])
