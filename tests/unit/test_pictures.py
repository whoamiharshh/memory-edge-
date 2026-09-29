"""Pictures as memory: stored as CLIP vectors in Qdrant Edge and found again by describing them.

The behaviour that matters is that nothing here describes a picture. Typed words retrieve photographs,
and the note a person wrote beside each one is what carries meaning — so these tests check retrieval and
check that a picture never becomes a citable claim.

The CLIP models are a ~590 MB download, so every test is skipped when they are not present rather than
failing a fresh clone.
"""
import pathlib

import pytest

pytest.importorskip("fastembed")
pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

from shared.image_embed import ImageEmbedder  # noqa: E402

pytestmark = pytest.mark.skipif(not ImageEmbedder().available, reason="fastembed not installed")


def _draw(path: pathlib.Path, kind: str) -> str:
    im = Image.new("RGB", (224, 224), "white")
    d = ImageDraw.Draw(im)
    if kind == "rust":
        d.rectangle((0, 0, 224, 224), fill=(196, 186, 170)); d.ellipse((40, 35, 185, 180), fill=(146, 68, 24))
    elif kind == "crack":
        d.rectangle((0, 0, 224, 224), fill=(168, 170, 176)); d.line((30, 195, 195, 25), fill=(18, 18, 18), width=10)
    else:
        d.rectangle((0, 0, 224, 224), fill=(238, 238, 240)); d.rectangle((50, 45, 175, 165), fill=(30, 110, 214))
    im.save(path)
    return str(path)


@pytest.fixture
def three_photos(tmp_path):
    return {k: _draw(tmp_path / f"{k}.png", k) for k in ("rust", "crack", "blue")}


def test_a_picture_is_found_by_describing_it(make_device, three_photos):
    """CLIP puts words and images in one space, so a photograph nobody labelled with the query's words
    can still be retrieved by them."""
    d, _ = make_device("devA", "site1")
    d.remember_image(three_photos["rust"], "pump housing flange")
    d.remember_image(three_photos["blue"], "guard fitted after the repair")

    hits = d.recall_images(text="rust and corrosion", limit=2)
    assert hits and "pump housing" in hits[0]["note_text"]


def test_a_picture_is_found_by_another_picture(make_device, three_photos, tmp_path):
    d, _ = make_device("devA", "site1")
    d.remember_image(three_photos["crack"], "hairline crack on the bracket")
    d.remember_image(three_photos["blue"], "new guard")

    similar = _draw(tmp_path / "similar.png", "crack")
    hits = d.recall_images(like_image=similar, limit=1)
    assert hits and "crack" in hits[0]["note_text"]


def test_a_picture_never_leaves_the_device(make_device, three_photos):
    """No outbox entry, and the record is marked local: a photograph has had no outcome verification and
    is not evidence about anything."""
    d, _ = make_device("devA", "site1")
    before = d.outbox.counts()
    d.remember_image(three_photos["rust"], "corrosion")
    assert d.outbox.counts() == before
    assert d.pictures()[0]["share_state"] == "local"


def test_pictures_ride_alongside_an_answer_and_are_never_cited(make_device, three_photos):
    """A photograph supports no sentence, so it must not appear as evidence the answer cites."""
    d, _ = make_device("devA", "site1")
    d.remember("The pump housing corroded badly last winter.")
    d.remember_image(three_photos["rust"], "corrosion on the pump housing")

    out = d.ask("what happened to the pump housing")
    assert out["grounded"] is True
    assert all(u["source"] != "picture" for u in out["used"]), "a picture is not citable evidence"
    assert out["pictures"], "a matching picture should still be offered alongside"


def test_unrelated_pictures_are_not_offered(make_device, three_photos):
    """Nearest-neighbour always returns something. A photo with nothing to do with the question is worse
    than none, because the answer presents them as matching."""
    d, _ = make_device("devA", "site1")
    d.remember("Deliveries arrive on Tuesday.")
    d.remember_image(three_photos["blue"], "new guard fitted after the repair")

    assert d.ask("when do deliveries arrive")["pictures"] == []


def test_a_missing_file_is_refused(make_device):
    d, _ = make_device("devA", "site1")
    with pytest.raises(ValueError):
        d.remember_image("no/such/photo.png", "note")


def test_pictures_survive_a_restart(make_device, three_photos):
    d, _ = make_device("devA", "site1")
    d.remember_image(three_photos["rust"], "corrosion on the flange")
    d.close()

    d2, _ = make_device("devA", "site1")
    assert any("corrosion" in p["note_text"] for p in d2.pictures())
