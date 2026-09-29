"""Shared free-text knowledge: one device publishes, only the addressed devices receive it, and they can
read it afterwards with the network off.

The central claim under test is that addressing is enforced by the cloud. A device must never be handed a
record meant for somebody else and merely told to hide it: a local copy can be read by whoever holds the
device, so filtering on the device would not be privacy at all.
"""
import pytest

from cloud import knowledge
from shared.embed import HashEmbedder


def hdr(tok):
    return {"Authorization": f"Bearer {tok}"}


@pytest.fixture
def three(server_cloud, make_server_device):
    """An author plus two recipients at different sites, all in one tenant."""
    reg, tenant = server_cloud["registry"], server_cloud["tenant"]
    devs = {}
    for name, site in (("author", "hq"), ("alice", "north"), ("bob", "south")):
        dev, worker = make_server_device(name, site, tenant)
        devs[name] = {"dev": dev, "worker": worker, "token": reg.issue(name, site, tenant)}
    return server_cloud, devs


def test_everyone_audience_reaches_every_device(three):
    cloud, devs = three
    devs["author"]["worker"].publish_knowledge("The lift is out of service today.", topic="notices")

    for who in ("alice", "bob"):
        assert devs[who]["worker"].pull_knowledge()["received"] == 1
        held = devs[who]["dev"].shared_knowledge()
        assert [h["note_text"] for h in held] == ["The lift is out of service today."]
        assert held[0]["author_device"] == "author"


def test_a_record_addressed_to_one_device_is_not_sent_to_the_others(three):
    cloud, devs = three
    devs["author"]["worker"].publish_knowledge("Alice's own record: reference 44.",
                                               audience="device", recipients=["alice"])

    assert devs["alice"]["worker"].pull_knowledge()["received"] == 1
    assert devs["bob"]["worker"].pull_knowledge()["received"] == 0
    assert devs["bob"]["dev"].shared_knowledge() == []


def test_the_cloud_never_hands_over_another_devices_record(three):
    """Checked at the API, not on the device: the bytes must not leave the server."""
    cloud, devs = three
    devs["author"]["worker"].publish_knowledge("Only for alice.", audience="device", recipients=["alice"])

    body = cloud["client"].get("/v1/knowledge", params={"since": 0},
                               headers=hdr(devs["bob"]["token"])).json()
    assert body["items"] == []


def test_site_audience_reaches_a_whole_site(three):
    cloud, devs = three
    devs["author"]["worker"].publish_knowledge("North site only.", audience="site", recipients=["north"])

    assert devs["alice"]["worker"].pull_knowledge()["received"] == 1     # alice is at "north"
    assert devs["bob"]["worker"].pull_knowledge()["received"] == 0       # bob is at "south"


def test_a_received_record_is_searchable_with_the_network_off(three):
    cloud, devs = three
    devs["author"]["worker"].publish_knowledge("Deliveries move to Wednesday from next month.")
    devs["alice"]["worker"].pull_knowledge()

    devs["alice"]["worker"].set_online(False)
    out = devs["alice"]["dev"].ask("when do deliveries happen")
    assert out["grounded"] is True
    assert out["used"][0]["source"] == "shared"


def test_withdrawing_removes_the_copy_on_the_recipient(three):
    """A withdrawn record is kept server-side with its text blanked, purely so the next pull tells the
    recipient to delete it. Dropping the row instead would leave every device that already pulled it
    answering from the record forever."""
    cloud, devs = three
    rec = devs["author"]["worker"].publish_knowledge("Temporary notice, will be cancelled.")
    devs["alice"]["worker"].pull_knowledge()
    assert len(devs["alice"]["dev"].shared_knowledge()) == 1

    assert knowledge.withdraw(cloud["store"], cloud["tenant"], rec["id"], "author") is True
    devs["alice"]["worker"].pull_knowledge()
    assert devs["alice"]["dev"].shared_knowledge() == []


def test_only_the_author_may_withdraw(three):
    cloud, devs = three
    rec = devs["author"]["worker"].publish_knowledge("Mine to withdraw.")
    assert knowledge.withdraw(cloud["store"], cloud["tenant"], rec["id"], "bob") is False
    assert knowledge.withdraw(cloud["store"], cloud["tenant"], rec["id"], "author") is True


def test_pull_is_incremental(three):
    cloud, devs = three
    devs["author"]["worker"].publish_knowledge("First.")
    assert devs["alice"]["worker"].pull_knowledge()["received"] == 1
    assert devs["alice"]["worker"].pull_knowledge()["received"] == 0, "nothing new to fetch"

    devs["author"]["worker"].publish_knowledge("Second.")
    assert devs["alice"]["worker"].pull_knowledge()["received"] == 1


def test_publishing_offline_is_refused(three):
    cloud, devs = three
    devs["author"]["worker"].set_online(False)
    with pytest.raises(RuntimeError, match="offline"):
        devs["author"]["worker"].publish_knowledge("Will not go anywhere.")


@pytest.mark.parametrize("kw", [
    {"audience": "device", "recipients": []},      # addressed to nobody
    {"audience": "nonsense"},
    {"text": ""},
])
def test_invalid_publishes_are_refused(three, kw):
    cloud, devs = three
    args = {"text": "some text", "audience": "everyone", "recipients": []} | kw
    with pytest.raises(ValueError):
        knowledge.publish(cloud["store"], HashEmbedder(), cloud["tenant"], "author", **args)
