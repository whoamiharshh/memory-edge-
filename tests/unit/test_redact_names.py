"""Names nobody put on the denylist (measured in bench/redaction.py on real logbook notes)."""
import pytest

from shared import name_model
from shared.redact import redact


@pytest.mark.parametrize("text", [
    "Bearing replaced by ravi, grease topped up",                  # lower case, on the name list
    "REPLACED FUEL SERVO, CHECKED BY SURESH",                      # all caps
    "Called Priya at the site office about the noise",             # mid-sentence capital
    "Motor realigned, Oluwaseun confirmed the readings",           # capitalised, unknown word
])
def test_unlisted_names_are_found(text):
    r = redact(text)
    assert not r.clean and "[REDACTED]" in r.text


@pytest.mark.parametrize("text", [
    "Inner race spall found on DE bearing 6205, replaced bearing, regreased, vibration back to normal",
    "COUPLING INSERT CRACKED. REPLACED INSERT, ALIGNED WITH LASER.",
    "Fan blade cleaned, imbalance gone after balancing on Monday",
    "Outer race wear, relubricated, noise gone",
])
def test_technical_notes_stay_clean(text):
    assert redact(text).clean


def test_a_name_after_a_cue_word_needs_only_a_lower_score():
    """'by / with / call ...' makes a name likelier: a word the model rates between 0.5 and its 0.9 threshold is
    caught after a cue, not elsewhere; a machine word after a cue (laser) stays clean."""
    from shared.redact import _lexicon
    m = name_model.load()
    lex = (frozenset(), _lexicon()[1])                       # no name list: only the model and the cue rule
    word = "okafor"
    assert 0.5 <= name_model.probability(m, [word])[0] < m["threshold"]
    cued = redact(f"REPLACED INSERT, CHECKED BY {word.upper()}", lexicon=lex, model=m)
    assert ("name_after_cue", word.upper()) in cued.findings
    uncued = redact(f"{word.upper()} REPLACED INSERT", lexicon=lex, model=m)
    assert all(k != "name_after_cue" for k, _ in uncued.findings)
    assert redact("ALIGNED WITH LASER, CHECKED WITH DIAL GAUGE", lexicon=lex, model=m).clean


def test_name_model_is_plain_numbers_with_a_validated_threshold():
    m = name_model.load()
    assert m and len(m["coef"]) == name_model.N_FEATURES and 0.5 <= m["threshold"] <= 0.95
    chosen = next(v for v in m["validation"] if v["threshold"] == m["threshold"])
    assert chosen["word_false_positive"] <= name_model.MAX_VAL_FP
    p = name_model.probability(m, ["venkatesh", "bearing", "coupling"])
    assert p[0] > 0.5 > max(p[1], p[2]) and max(p[1], p[2]) < m["threshold"]   # a name scores far above machine words
