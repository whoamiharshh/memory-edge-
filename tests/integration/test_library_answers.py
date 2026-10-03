"""The four failures seen on screen on 2 Oct (night) and the model fallback the user asked for.

  "capital of tamilnadu"            -> no answer (the library says "Tamil Nadu", two words)
  "what is dna"                     -> Sleeping Beauty transposon / RNA instead of the DNA article
  "how many states does India have" -> a cached page about the Great Wall of China, matched on the word "states"
  "what is the capital of France"   -> "From what this device holds", plus a second answer about Nouvelle-Aquitaine

and: the local model answers only simple general questions, labelled unverified, and says it does not know when unsure.
"""
import pytest

from edge import rag, seed
from edge.device import _covered, _simple_general


def _lib(dev, *rows):
    """Put library entries on the device the way edge/seed.py does (title, text, topic)."""
    batch = [{"ref_id": rid, "title": title, "text": text, "sources": ["https://example.org/" + rid],
              "topic": topic, "pack": "test"} for rid, title, text, topic in rows]
    seed._flush(dev, batch)


TAMIL = ("wiki:Tamil Nadu", "Tamil Nadu", "Tamil Nadu: Tamil Nadu is a state in the South of India. The capital of "
         "this state is Chennai. Other large cities in Tamil Nadu include Coimbatore, Salem and Madurai.",
         "Simple English Wikipedia")
DNA = ("wiki:DNA", "DNA", "DNA: DNA, short for deoxyribonucleic acid, is the molecule that contains the genetic code "
       "of living organisms.", "Simple English Wikipedia")
GENOME = ("wiki:Genome", "Genome", "Genome: The genome of an organism is the whole of its hereditary information "
          "encoded in its DNA (or, for some viruses, RNA).", "Simple English Wikipedia")
RNA = ("wiki:RNA", "RNA", "RNA: RNA is physically different from DNA. DNA has two strands.", "Simple English Wikipedia")
FRANCE_FACT = ("capital:Q142", "Capital of France", "The capital of France is Paris.", "general knowledge")
NOUVELLE = ("wiki:Nouvelle-Aquitaine", "Nouvelle-Aquitaine", "Nouvelle-Aquitaine: Nouvelle-Aquitaine is one of the "
            "administrative regions of France. Its capital is Bordeaux.", "Simple English Wikipedia")


@pytest.fixture
def dev(make_device):
    d, _ = make_device("solo", "site1")
    d.outbox.kv_set("online", False)
    return d


def test_place_names_match_with_or_without_the_space(dev):
    assert _covered({"capital", "tamilnadu"}, TAMIL[2]) == {"capital", "tamilnadu"}
    _lib(dev, TAMIL)
    out = dev.ask("capital of tamilnadu", None)
    assert out["grounded"] is True and out["sources"] == ["offline_kb"]
    assert "Chennai" in out["answer"]


def test_asking_what_is_x_answers_with_the_article_named_x_only(dev):
    _lib(dev, GENOME, RNA, DNA)
    out = dev.ask("what is dna", None)
    assert [u["title"] for u in out["used"]] == ["DNA"]
    assert "deoxyribonucleic" in out["answer"]


def test_a_cached_web_page_cannot_answer_on_one_shared_word(dev):
    dev.remember("Great Wall of China: The Great Wall is a series of fortifications in China. In the United States "
                 "it is a famous landmark.", kind="learned", meta={"url": "https://x.org/gw", "title": "Great Wall"})
    out = dev.ask("How many states does India have?", None)
    assert out["mode"] == "no_evidence" and out["used"] == []


def test_the_library_is_never_called_what_this_device_holds(dev):
    _lib(dev, FRANCE_FACT, NOUVELLE)
    out = dev.ask("What is the capital of France?", None)
    assert out["sources"] == ["offline_kb"]
    assert "this device holds" not in out["answer"]
    assert out["answer"].startswith("The capital of France is Paris.")
    assert [u["title"] for u in out["used"]] == ["Capital of France"]          # no second, unrelated answer


def test_device_notes_still_say_what_this_device_holds(dev):
    dev.remember("The spare bearing kit is kept in cabinet 7.", kind="fact")
    out = dev.ask("where is the spare bearing kit kept", None)
    assert out["answer"].startswith("From what this device holds:") and out["sources"] == ["device"]


def test_title_lookup_names_the_longest_phrase_only():
    ids = [rid for rid, _ in seed.title_ref_ids("what is the capital of tamilnadu")]
    assert ids and all(r.startswith("india:") for r in ids)        # the fact "Capital of Tamil Nadu", nothing shorter
    assert "wiki:Capital" not in ids and "wiki:Tamil Nadu" not in ids
    assert "wiki:Tamil Nadu" in [r for r, _ in seed.title_ref_ids("tell me about tamilnadu")]
    assert "wiki:DNA" in [r for r, _ in seed.title_ref_ids("what is dna")]


# ---- the local model: simple general questions only, unverified, "I don't know" when unsure -----------------------

class Scripted:
    """Stands in for LocalLLM: replays replies, records how often it was asked."""
    available = True

    def __init__(self, *replies):
        import threading
        self.replies, self.calls, self._lock = list(replies), 0, threading.Lock()

    def _load(self):
        return self

    def create_chat_completion(self, messages, **kw):
        r = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return {"choices": [{"message": {"content": r}}]}


class Forbidden:
    available = True

    def __getattr__(self, name):
        raise AssertionError(f"the model must not be asked: .{name}")


def test_model_answers_a_simple_general_question_and_labels_it_unverified(dev):
    llm = Scripted("The Pacific is the largest ocean on Earth.")
    out = dev.ask("Which is the largest ocean?", llm)
    assert out["mode"] == "model_unverified" and out["unverified"] is True and out["grounded"] is False
    assert out["answer"].startswith("[Unverified]") and "Pacific" in out["answer"]
    assert out["sources"] == [] and out["used"] == [] and "Unverified" in out["label"]
    assert llm.calls == (2 if rag.is_big_model() else 3)   # greedy once, then again to check it agrees with itself


def test_model_that_changes_its_answer_is_treated_as_guessing(dev):
    out = dev.ask("Which is the largest ocean?",
                  Scripted("The Pacific is the largest ocean.", "The Atlantic is the largest ocean.", "Indian."))
    assert out["mode"] == "no_evidence" and out.get("model_unsure") is True
    assert out["answer"] == "Needs internet connection for this." and "unverified" not in out


def test_model_that_says_it_does_not_know_is_not_dressed_up_as_an_answer(dev):
    out = dev.ask("Who was the first emperor of Rome?", Scripted("I DON'T KNOW"))
    assert out["mode"] == "no_evidence" and out["answer"] == "Needs internet connection for this."


@pytest.mark.parametrize("q", [
    "What causes bearing vibration at 2x running speed?",
    "Why does my pump keep failing?",
    "What is wrong with this device?",
    "How do I fix error P0301?",
    "what does the fleet say about gearbox noise",
])
def test_machine_questions_never_reach_the_model(dev, q):
    out = dev.ask(q, Forbidden())
    assert out["mode"] == "no_evidence" and out["grounded"] is False and "unverified" not in out


def test_the_model_is_not_asked_when_a_source_answers(dev):
    _lib(dev, FRANCE_FACT)
    out = dev.ask("What is the capital of France?", Forbidden())
    assert out["grounded"] is True and out["sources"] == ["offline_kb"]


def test_a_question_that_depends_on_when_or_where_never_reaches_the_model(dev):
    out = dev.ask("Who won the cricket match yesterday?", Forbidden())
    assert out["mode"] == "no_evidence" and "unverified" not in out


def test_simple_general_classifier():
    assert _simple_general("Who wrote Hamlet?") and _simple_general("How many legs does a spider have?")
    assert not _simple_general("how do I repair a bearing")
    assert not _simple_general("what is the status of plot 91")
    assert not _simple_general("explain quantum chromodynamics and its relation to confinement in great detail please now")
    assert not _simple_general("tell me a joke")
    for q in ("What is the phone number of the nearest pharmacy?", "What was the temperature in Delhi yesterday?",
              "What is the wifi password of the office?", "What is the latest iPhone?"):
        assert not _simple_general(q), q


def test_general_answer_rejects_a_model_that_only_repeats_the_question():
    assert rag.general_answer(Scripted("Which ocean is the largest?"), "Which ocean is the largest?") is None


# ---- the extractor bug behind the corrupted DNA lead ---------------------------------------------------------------

def test_image_caption_with_a_link_inside_is_removed_whole():
    from tools.extract_simplewiki import clean
    wikitext = ("[[File:DNA.png|thumb|upright=1.4|The structure of [[DNA]]. The phosphate groups are yellow, the "
                "atoms are: P=phosphorus]]\n'''DNA''', short for deoxyribonucleic acid, is the molecule that contains "
                "the genetic code of living organisms. This includes animals, plants and bacteria.\n")
    out = clean(wikitext)
    assert out.startswith("DNA, short for deoxyribonucleic acid") and "phosphate" not in out


# ---- a library fix has to reach devices that already loaded the old files -----------------------------------------

def _pack_rows(dev, pack):
    return dev.store.count({"type": "reference", "device_id": dev.cfg.device_id, "source_pack": pack})


def test_a_changed_shipped_library_is_reloaded_over_the_old_copy(dev):
    seed._flush(dev, [{"ref_id": "wiki:DNA", "title": "DNA", "text": "DNA: of DNA. The phosphate groups are yellow.",
                       "sources": [], "topic": "Simple English Wikipedia", "pack": "simple_wikipedia"}])
    dev.outbox.kv_set(seed.SEED_FLAG, ["simple_wikipedia", "vehicle_codes"])
    dev.outbox.kv_set(seed.VERSION_FLAG, "an-older-version")
    todo = seed.refresh_if_stale(dev)
    assert "simple_wikipedia" in todo and _pack_rows(dev, "simple_wikipedia") == 0
    assert seed.seeded(dev) == ["vehicle_codes"]                  # what the person opted into by hand stays loaded
    assert dev.outbox.kv_get(seed.VERSION_FLAG) == seed.library_version()


def test_an_unchanged_library_is_left_alone(dev):
    seed._flush(dev, [{"ref_id": "wiki:DNA", "title": "DNA", "text": "DNA: a molecule.", "sources": [],
                       "topic": "Simple English Wikipedia", "pack": "simple_wikipedia"}])
    dev.outbox.kv_set(seed.SEED_FLAG, ["simple_wikipedia"])
    dev.outbox.kv_set(seed.VERSION_FLAG, seed.library_version())
    assert "simple_wikipedia" not in seed.refresh_if_stale(dev) and _pack_rows(dev, "simple_wikipedia") == 1


def test_a_curated_fact_is_the_whole_answer_not_one_of_its_look_alikes(dev):
    _lib(dev, ("india:count", "Number of states and union territories of India",
               "India has 28 states and 8 union territories.", "general knowledge"),
         ("india:Q1", "Capital of Rajasthan", "The capital of Rajasthan, a state of India, is Jaipur.",
          "general knowledge"))
    out = dev.ask("How many states does India have?", None)
    assert [u["title"] for u in out["used"]] == ["Number of states and union territories of India"]


def test_this_device_does_not_make_a_new_question_a_follow_up():
    from edge.device import _is_follow_up
    assert not _is_follow_up("What is wrong with this device?")
    assert _is_follow_up("how does this work?") and _is_follow_up("why does it vibrate") and _is_follow_up("and then?")


def test_a_curated_fact_must_cover_every_word_of_the_question(dev):
    """'national animal of India' shared two of its three words with the fact about the National Capital Territory
    of Delhi and was answered with it. The word that mattered, 'animal', was nowhere in the fact."""
    _lib(dev, ("india:Q1498", "Capital of National Capital Territory of Delhi",
               "The capital of National Capital Territory of Delhi, a union territory of India, is New Delhi.",
               "general knowledge"))
    out = dev.ask("What is the national animal of India?", None)
    assert out["mode"] == "no_evidence" and out["used"] == []


def test_filler_words_do_not_stop_a_fact_from_answering(dev):
    _lib(dev, FRANCE_FACT)
    out = dev.ask("What is the capital city of France?", None)
    assert out["grounded"] is True and "Paris" in out["answer"]


def test_when_it_does_not_know_and_is_offline_it_says_it_needs_the_internet(dev):
    for q in ("Who was the first emperor of Rome?", "What is the wifi password of the office?",
              "Design a fault tolerant production line"):
        out = dev.ask(q, None)
        assert out["answer"] == "Needs internet connection for this." and out["needs_internet"] is True, q
        assert out["grounded"] is False and out["sources"] == []


# ---- the technical library: aliases, comparisons, and questions that have nothing to quote ----------------------------

PLC = ("tech:Programmable logic controller", "Programmable logic controller",
       "Programmable logic controller: A programmable logic controller (PLC) is an industrial computer that has been "
       "ruggedized and adapted for the control of manufacturing processes.", "Technical concepts (Wikipedia)",
       ["PLC", "Programmable logic controller"])
DCS = ("tech:Distributed control system", "Distributed control system",
       "Distributed control system: A distributed control system (DCS) is a computerized control system for a process "
       "or plant usually with many control loops.", "Technical concepts (Wikipedia)",
       ["DCS", "Distributed control system"])


def _tech(dev, *rows):
    seed._flush(dev, [{"ref_id": r[0], "title": r[1], "text": r[2], "sources": ["https://en.wikipedia.org/wiki/x"],
                       "topic": r[3], "pack": "technical", "aliases": r[4]} for r in rows])


def test_a_term_is_found_by_the_name_people_type(dev):
    _tech(dev, PLC)
    out = dev.ask("What is a PLC?", None)
    assert out["sources"] == ["offline_kb"] and "industrial computer" in out["answer"]
    assert [u["title"] for u in out["used"]] == ["Programmable logic controller"]


def test_how_does_x_work_returns_the_lead_for_x(dev):
    _tech(dev, PLC)
    out = dev.ask("How does a PLC work?", None)
    assert out["grounded"] is True and "industrial computer" in out["answer"]


def test_x_versus_y_shows_both_definitions_and_says_it_is_not_a_comparison(dev):
    _tech(dev, PLC, DCS)
    out = dev.ask("PLC vs DCS?", None)
    assert {u["title"] for u in out["used"]} == {"Programmable logic controller", "Distributed control system"}
    assert out["answer"].startswith("The library holds each definition, not a side-by-side comparison")


@pytest.mark.parametrize("q", ["Design a PLC controlled packaging line", "What if the PLC loses power?",
                               "How would you diagnose a PLC that does not execute?", "Calculate PLC scan time",
                               "Why use PLCs instead of PCs?"])
def test_design_calculation_and_what_if_questions_are_not_answered_with_a_lead(dev, q):
    _tech(dev, PLC)
    out = dev.ask(q, Forbidden())
    assert out["answer"] == "Needs internet connection for this." and out["used"] == []


def test_the_shipped_technical_library_defines_the_terms_in_the_question_list():
    import gzip
    import json
    from edge.seed import KNOWLEDGE
    rows = [json.loads(line) for line in gzip.open(KNOWLEDGE / "technical_wikipedia.jsonl.gz", "rt", encoding="utf-8")]
    names = {a.lower() for r in rows for a in r["aliases"]}
    for term in ("plc", "scada", "pid controller", "kalman filter", "jacobian", "can bus", "modbus", "ros", "adas", "slam"):
        assert term in names, term
    assert all(r["url"].startswith("https://en.wikipedia.org/wiki/") and len(r["text"]) >= 60 for r in rows)


def test_an_alias_that_points_at_two_articles_is_dropped_not_guessed():
    import gzip
    import json
    from edge.seed import KNOWLEDGE
    rows = [json.loads(line) for line in gzip.open(KNOWLEDGE / "technical_wikipedia.jsonl.gz", "rt", encoding="utf-8")]
    owner: dict[str, set] = {}
    for r in rows:
        for a in r["aliases"]:
            owner.setdefault("".join(ch for ch in a.lower() if ch.isalnum()), set()).add(r["title"])
    assert [k for k, v in owner.items() if len(v) > 1] == []


# ---- small talk: "hi" is not a question --------------------------------------------------------------------------

@pytest.mark.parametrize("msg,kind", [("hi", "greeting"), ("Hello!", "greeting"), ("hey there", "greeting"),
                                      ("thanks", "thanks"), ("bye", "bye"), ("how are you?", "how_are_you"),
                                      ("who are you", "identity"), ("what can you do", "help")])
def test_small_talk_gets_a_direct_reply_and_never_touches_retrieval(dev, msg, kind):
    out = dev.ask(msg, Forbidden())
    assert out["mode"] == "smalltalk" and out["grounded"] is True and out["used"] == [] and out["sources"] == []
    assert "Needs internet" not in out["answer"]


def test_a_greeting_after_a_question_does_not_inherit_the_old_topic(dev):
    _lib(dev, TAMIL)
    dev.ask("capital of tamilnadu", None)
    out = dev.ask("hi", None)
    assert out["mode"] == "smalltalk" and "Chennai" not in out["answer"]


def test_a_greeting_with_a_real_question_is_still_a_question(dev):
    _lib(dev, TAMIL)
    out = dev.ask("hi what is the capital of tamilnadu", None)
    assert out["mode"] != "smalltalk"


def test_hindi_greeting_gets_a_hindi_reply(dev):
    out = dev.ask("नमस्ते", None)
    assert out["mode"] == "smalltalk" and out["language"] == "hi" and "नमस्ते" in out["answer"]


@pytest.mark.parametrize("q,expect", [("what is 12 times 7", "12 * 7 = 84"), ("2+2*3", "2+2*3 = 8"),
                                      ("what's 100 divided by 8", "100 / 8 = 12.5"), ("what is 2 to the power of 10", "2 ^ 10 = 1024")])
def test_arithmetic_is_calculated_exactly_not_asked_of_a_model(dev, q, expect):
    out = dev.ask(q, Forbidden())
    assert out["mode"] == "calculated" and out["answer"] == expect


def test_a_question_with_a_plain_number_can_still_reach_the_model(dev):
    from edge.device import _simple_general
    assert _simple_general("Does water boil at 100 degrees Celsius?") and _simple_general("Tell me a fact about the sun")
    assert not _simple_general("what is the status of plot91")


def test_declining_is_just_the_one_sentence(dev):
    dev.remember("The pump house key is in the blue cabinet.", kind="fact")
    out = dev.ask("who was the first person on mars", None)
    assert out["answer"] == "Needs internet connection for this."


def test_a_passage_about_another_entity_is_not_an_answer(dev):
    """'first prime minister of India' was answered with Tunisia's first prime minister."""
    _lib(dev, ("wiki:Prime Minister of Tunisia", "Prime Minister of Tunisia",
               "Prime Minister of Tunisia: Mustapha Dinguizli was Tunisia's first Prime Minister. He served in 1922.",
               "Simple English Wikipedia"))
    out = dev.ask("who is the first prime minister of india", None)
    assert out["used"] == [] and out["answer"] == "Needs internet connection for this."


def test_generic_words_do_not_pull_in_device_records(dev):
    dev.remember("Local episode on this machine: bearing inner race, the likely cause is wear.", kind="fact")
    out = dev.ask("what causes rain", None)
    assert out["used"] == [] and out["answer"] == "Needs internet connection for this."
    out = dev.ask("what is machine learning", None)
    assert out["used"] == []
