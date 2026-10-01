"""Hindi <-> English: language detection, the hand-written fixed sentences, and that nothing a translation
must never touch (evidence markers, ids, numbers) is altered. The model-backed tests skip when the converted
models are not on this machine; the others need no model."""
import pytest

from shared import translate as T

needs_models = pytest.mark.skipif(not T.available("hi"), reason="translation models not built")


def test_detects_devanagari_as_hindi_and_latin_as_english():
    assert T.detect("मैंने कौन सी समस्याएँ देखी हैं?") == "hi"
    assert T.detect("What problems have you seen?") == "en"
    assert T.detect("") == "en"
    assert T.detect("1234 !!") == "en"


def test_a_few_hindi_words_in_an_english_sentence_stay_english():
    assert T.detect("What does the word बेयरिंग mean in this manual and where is it used on motor 1?") == "en"


def test_fixed_device_sentences_use_the_hand_written_hindi_without_any_model():
    en = "Nothing on this device relates to that. This device is offline, so it cannot look it up."
    hi = T.from_english(en, "hi")
    assert hi == (T.FIXED_HI["Nothing on this device relates to that."] + " "
                  + T.FIXED_HI["This device is offline, so it cannot look it up."])
    assert T.detect(hi) == "hi"


def test_the_episode_overview_is_hand_written_hindi_with_every_stored_value_intact():
    en = ("1 episode(s) have been opened on this machine.\n\n"
          "Episode #1 on devA-motor1: bearing / inner_race (confirmed by a technician), 40 window(s), first "
          "seen 2026-10-01T07:54. Action taken: replace_bearing (outcome recorded as worked). The sensor then "
          "saw 20 consecutive healthy windows: symptom resolved. [E1]")
    hi = T.from_english(en, "hi")
    assert "इस मशीन पर 1 एपिसोड दर्ज हुए हैं।" in hi
    assert "एपिसोड #1, devA-motor1: bearing / inner_race (तकनीशियन द्वारा पुष्टि), 40 विंडो" in hi
    assert "की गई कार्रवाई: replace_bearing (नतीजा दर्ज: सफल)।" in hi
    assert "लगातार 20 सामान्य विंडो देखीं: लक्षण ठीक हो गया।" in hi
    assert hi.rstrip().endswith("[E1]") and "2026-10-01T07:54" in hi
    assert "symptom" not in hi and "Episode" not in hi        # nothing left in English to be mistranslated


def test_the_whole_record_question_in_hindi_reaches_the_right_english_question():
    from edge.device import _overview_intent
    asked = {"क्या यहाँ कुछ ठीक किया गया है?": "fixed", "मैंने कौन सी समस्याएँ देखी हैं?": "problems",
             "आप मेरे बारे में क्या जानते हैं?": "known", "क्या कुछ अब भी अनसुलझा है?": "unresolved"}
    for hi_q, intent in asked.items():
        assert _overview_intent(T.to_english(hi_q, "hi")) == intent, hi_q


def test_english_passes_through_untouched():
    assert T.to_english("anything", "en") == "anything"
    assert T.from_english("anything [E1]", "en") == "anything [E1]"


@needs_models
def test_hindi_question_translates_to_plain_english():
    assert "capital of india" in T.to_english("भारत की राजधानी क्या है?", "hi").lower()


@needs_models
def test_markers_ids_and_numbers_survive_translation_back_to_hindi():
    en = ("Episode #1 on devA-motor1: bearing / inner_race, 40 windows, first seen 2026-10-01T07:54. "
          "Action taken: replace_bearing. The sensor then saw 20 healthy windows. [E1]")
    hi = T.from_english(en, "hi")
    for kept in ("devA-motor1", "inner_race", "2026-10-01T07:54", "replace_bearing", "[E1]", "40", "20"):
        assert kept in hi, f"{kept!r} was lost: {hi}"
    assert T.detect(hi) == "hi"
