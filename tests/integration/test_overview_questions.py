"""Questions about the device's record as a whole ("what problems have you seen?").

These are the four questions the device UI offers as starting points, so they are the first thing anyone
asks. They name nothing that appears in a stored record, so the word-overlap relevance test that answers
every other question finds nothing for them: before this path existed, a device holding a technician-
confirmed, sensor-verified repair answered "nothing on this device relates to that at all".

The answers are built from stored episode fields, never from retrieval, and they must stay inside the
project's wording rule: a sensor resolves a symptom, it does not confirm a root cause, and nothing here
recommends an action.
"""
import pytest

from tests.conftest import needs_cwru
from tests.integration.test_fleet_flow import feed, resolve_episode, setup_machine

pytestmark = needs_cwru

STARTERS = ["What problems have you seen recently?", "Has anything here been fixed before?",
            "Is anything still unresolved?", "What do you know about me?"]


# ---- an empty device says what is true, not "that is outside my world" -----------------------------------
@pytest.mark.parametrize("q", STARTERS)
def test_starter_questions_on_an_empty_device_are_answered_honestly(make_device, q):
    d, _ = make_device("devA", "site1")
    out = d.ask(q)
    assert out["mode"] == "overview"
    assert out["used"] == []
    # the generic refusal would be wrong here: the device understood the question perfectly, it simply
    # has nothing recorded yet, and those two need different things from the person
    assert "relates to that at all" not in out["answer"]
    # it opens by saying it holds none of the thing asked about ("Nothing has been flagged...",
    # "No repair has been recorded...") rather than by refusing the question
    assert out["answer"].lower().startswith(("nothing", "no ", "this device holds nothing"))


# ---- with a real verified repair on board ----------------------------------------------------------------
@pytest.fixture
def repaired(make_device):
    """A device carrying one technician-confirmed, sensor-verified repair (CWRU 105 -> healthy 99)."""
    d, _ = make_device("devA", "site1")
    setup_machine(d)
    eid = resolve_episode(d, 105, "replace_bearing", "inner_race", note="Inner race spall, bearing replaced")
    assert d.episode(eid)["verify"]["verdict"] == "symptom_resolved"
    return d


@pytest.mark.parametrize("q", ["Has anything here been fixed before?", "any repairs?", "what worked here"])
def test_a_verified_repair_is_found_by_asking_whether_anything_was_fixed(repaired, q):
    out = repaired.ask(q)
    assert out["mode"] == "overview" and out["grounded"] is True
    assert len(out["used"]) == 1, "the one recorded repair should be the one piece of evidence"
    assert "replace_bearing" in out["answer"]
    assert "symptom resolved" in out["answer"]
    assert "[E1]" in out["answer"] and out["used"][0]["key"] == "E1"   # cited, so the UI can show its source


def test_it_never_claims_the_sensor_confirmed_a_root_cause(repaired):
    for q in STARTERS:
        answer = repaired.ask(q)["answer"].lower()
        assert "root cause" not in answer
        for banned in ("you should", "we recommend", "recommended action"):
            assert banned not in answer


def test_what_problems_lists_the_episode_with_its_confirmed_class(repaired):
    out = repaired.ask("What problems have you seen recently?")
    assert out["mode"] == "overview" and out["grounded"] is True
    assert "inner_race" in out["answer"]
    assert "confirmed by a technician" in out["answer"], "a confirmed class must not read as a guess"


def test_a_closed_and_verified_episode_is_not_reported_as_outstanding(repaired):
    out = repaired.ask("Is anything still unresolved?")
    assert out["mode"] == "overview"
    assert out["used"] == [] and "Nothing is outstanding" in out["answer"]


def test_an_unverified_episode_is_reported_as_outstanding(make_device):
    """An episode whose repair has not been confirmed must read as outstanding, never as fixed."""
    d, _ = make_device("devA", "site1")
    setup_machine(d)
    feed(d, 105, n=25)                                   # a fault, with no action recorded against it
    assert d.episodes()

    unresolved = d.ask("Is anything still unresolved?")
    assert unresolved["grounded"] is True and len(unresolved["used"]) == 1
    assert "No action has been recorded against it yet" in unresolved["answer"]

    fixed = d.ask("Has anything here been fixed before?")
    assert fixed["used"] == [] and "No repair has been recorded" in fixed["answer"]


def test_what_do_you_know_reports_totals_from_what_is_stored(repaired):
    repaired.remember("The spare bearings live in the blue cabinet.")
    out = repaired.ask("What do you know about me?")
    assert out["mode"] == "overview"
    assert "devA-m1" in out["answer"] and "site1" in out["answer"]
    assert "1 episode(s)" in out["answer"] and "1 thing(s) you taught it" in out["answer"]


# ---- the overview must not swallow questions about one specific thing -------------------------------------
def test_a_question_naming_a_specific_thing_still_uses_retrieval(repaired):
    """"Has plot 91 been fixed?" names its own subject; answering that with a summary of everything would
    be wrong, so an identifier in the question sends it back down the retrieval path."""
    assert repaired.ask("has plot 91 been fixed?")["mode"] != "overview"


@pytest.mark.parametrize("q", ["what happened to the pump housing", "what went wrong with the gearbox",
                               "what problems does the bearing have"])
def test_a_question_that_names_its_subject_is_not_an_overview(repaired, q):
    """The phrasing alone is not enough: "what happened to the pump housing" shares "what happened" with
    "what happened recently", and only the subject it names tells them apart. Naming one sends it to
    retrieval, where the record about that subject is what answers it."""
    assert repaired.ask(q)["mode"] != "overview"


def test_an_ordinary_question_is_unaffected(repaired):
    repaired.remember("The spare keys are in the second drawer of the blue cabinet.")
    out = repaired.ask("where are the spare keys kept")
    assert out["mode"] != "overview" and out["grounded"] is True
    assert "second drawer" in out["answer"]
