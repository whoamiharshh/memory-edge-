"""Online retrieval, and the rule deciding when a question may leave the device.

Nothing here reaches the internet. The providers are exercised against recorded HTML/JSON, and the
routing is exercised with the search function stubbed, because a test that depends on a search engine
being up cannot tell anyone whether this code is correct.
"""
import httpx
import pytest

from edge import online


@pytest.fixture
def provider_env(monkeypatch):
    def _set(name: str, **env):
        monkeypatch.setenv("EDGE_SEARCH_PROVIDER", name)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
    return _set


# ---- what the UI is told ---------------------------------------------------------------------------------
def test_default_provider_needs_no_key(monkeypatch):
    monkeypatch.delenv("EDGE_SEARCH_PROVIDER", raising=False)
    a = online.available()
    assert a["provider"] == "duckduckgo" and a["ready"] is True and a["needs_key"] is None


def test_a_keyed_provider_is_not_ready_until_its_key_is_set(provider_env, monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    provider_env("brave")
    assert online.available() == {"provider": "brave", "enabled": True,
                                  "needs_key": "BRAVE_SEARCH_API_KEY",
                                  "key_present": False, "ready": False}
    provider_env("brave", BRAVE_SEARCH_API_KEY="k")
    assert online.available()["ready"] is True


def test_none_disables_retrieval_entirely(provider_env):
    provider_env("none")
    assert online.available()["enabled"] is False
    assert online.search("anything at all") == []


# ---- providers, against recorded responses ---------------------------------------------------------------
DDG_HTML = """
<div><a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fai">Latest <b>AI</b> news</a>
<a class="result__snippet">Models shipped this week.</a></div>
<div><a class="result__a" href="https://direct.example.org/page">Direct link</a>
<a class="result__snippet">No redirect wrapper here.</a></div>
"""


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x")


def test_duckduckgo_unwraps_the_redirect_and_strips_markup():
    hits = online._duckduckgo("ai", 5, _client(lambda r: httpx.Response(200, text=DDG_HTML)))
    assert [h["url"] for h in hits] == ["https://example.com/ai", "https://direct.example.org/page"]
    assert hits[0]["title"] == "Latest AI news", "tags inside the title must not survive"
    assert hits[0]["snippet"] == "Models shipped this week."


def test_wikipedia_returns_the_intro_extract_and_a_real_url():
    def handler(request):
        if request.url.params.get("list") == "search":
            return httpx.Response(200, json={"query": {"search": [{"title": "Edge computing",
                                                                   "snippet": "fallback"}]}})
        return httpx.Response(200, json={"query": {"pages": {"1": {"title": "Edge computing",
                                                                  "extract": "Edge computing is a model."}}}})
    hits = online._wikipedia("edge computing", 3, _client(handler))
    assert hits == [{"title": "Edge computing",
                     "url": "https://en.wikipedia.org/wiki/Edge_computing",
                     "snippet": "Edge computing is a model."}]


def test_a_provider_that_fails_never_raises(monkeypatch):
    """An unreachable network is this product's normal condition, not an exception."""
    def boom(*a):
        raise httpx.ConnectError("down")
    monkeypatch.setattr(online, "_duckduckgo", boom)
    monkeypatch.setattr(online, "_wikipedia", boom)
    monkeypatch.setattr(online, "_PROVIDERS", {"duckduckgo": [boom]})
    monkeypatch.setenv("EDGE_SEARCH_PROVIDER", "duckduckgo")
    assert online.search("anything") == []


def test_results_are_deduplicated_by_url(monkeypatch):
    same = [{"title": "A", "url": "https://e.com/a", "snippet": "x"}]
    monkeypatch.setattr(online, "_PROVIDERS",
                        {"duckduckgo": [lambda *a: list(same), lambda *a: list(same)]})
    monkeypatch.setenv("EDGE_SEARCH_PROVIDER", "duckduckgo")
    assert len(online.search("a", limit=5)) == 1


# ---- when a question is allowed to leave the device ------------------------------------------------------
def test_local_mode_never_goes_online(make_device, monkeypatch):
    monkeypatch.setenv("EDGE_SEARCH_PROVIDER", "duckduckgo")
    d, _ = make_device("devA", "site1")
    d.set_retrieval_mode("local")
    assert d._may_go_online(has_local_answer=False) == (False, "set to this device only")


def test_an_offline_device_never_goes_online(make_device, monkeypatch):
    monkeypatch.setenv("EDGE_SEARCH_PROVIDER", "duckduckgo")
    d, _ = make_device("devA", "site1")
    d.set_retrieval_mode("online")
    d.outbox.kv_set("online", False)
    assert d._may_go_online(has_local_answer=False) == (False, "offline")


def test_auto_prefers_the_device_when_it_already_holds_the_answer(make_device, monkeypatch):
    monkeypatch.setenv("EDGE_SEARCH_PROVIDER", "duckduckgo")
    d, _ = make_device("devA", "site1")
    assert d.retrieval_mode() == "auto", "auto is the default"
    assert d._may_go_online(has_local_answer=True)[0] is False
    assert d._may_go_online(has_local_answer=False)[0] is True


def test_online_mode_goes_out_even_when_the_device_has_something(make_device, monkeypatch):
    monkeypatch.setenv("EDGE_SEARCH_PROVIDER", "duckduckgo")
    d, _ = make_device("devA", "site1")
    d.set_retrieval_mode("online")
    assert d._may_go_online(has_local_answer=True)[0] is True


def test_the_mode_survives_a_restart(make_device):
    from edge.device import Device, DeviceConfig
    from shared.embed import HashEmbedder
    d, _ = make_device("devA", "site1")
    d.set_retrieval_mode("local")
    root = d.cfg.root
    d.close()
    again = Device(DeviceConfig(device_id="devA", site_id="site1", machine_id="devA-m1", root=root),
                   HashEmbedder())
    try:
        assert again.retrieval_mode() == "local"
    finally:
        again.close()


def test_a_bad_mode_is_refused(make_device):
    d, _ = make_device("devA", "site1")
    with pytest.raises(ValueError):
        d.set_retrieval_mode("whatever")


# ---- the answer says where it came from ------------------------------------------------------------------
def test_a_web_answer_is_labelled_as_web_and_carries_its_url(make_device, monkeypatch):
    monkeypatch.setenv("EDGE_SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setattr(online, "search",
                        lambda q, limit=5: [{"title": "Mars", "url": "https://e.org/mars",
                                             "snippet": "Mars is the fourth planet."}])
    d, _ = make_device("devA", "site1")
    out = d.ask("what is mars")

    assert out["sources"] == ["web"] and out["grounded"] is True
    assert out["used"][0]["source"] == "web" and out["used"][0]["url"] == "https://e.org/mars"
    # the lead must not claim the device holds what it just read on the internet
    assert "From the internet" in out["answer"]
    assert "From what this device holds" not in out["answer"]


def test_a_local_answer_still_says_it_came_from_the_device(make_device, monkeypatch):
    monkeypatch.setenv("EDGE_SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setattr(online, "search", lambda q, limit=5: [{"title": "no", "url": "https://e.org/x",
                                                               "snippet": "irrelevant"}])
    d, _ = make_device("devA", "site1")
    d.remember("The spare keys are in the second drawer of the blue cabinet.")
    out = d.ask("where are the spare keys kept")
    assert out["sources"] == ["device"]
    assert "From what this device holds" in out["answer"]


def test_a_web_hit_sharing_no_word_with_the_question_is_not_cited(make_device, monkeypatch):
    """A search engine always returns something. "Norfolk State University" came back for a question
    about the president of France, and citing it would be worse than citing nothing."""
    monkeypatch.setenv("EDGE_SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setattr(online, "search", lambda q, limit=5: [
        {"title": "President of France", "url": "https://e.org/fr", "snippet": "The French president."},
        {"title": "Norfolk State University", "url": "https://e.org/nsu", "snippet": "A university."}])
    d, _ = make_device("devA", "site1")
    out = d.ask("who is the current president of France")
    assert [u["url"] for u in out["used"]] == ["https://e.org/fr"]


# ---- what it looked up once, it knows with the network off ------------------------------------------------
HIT = [{"title": "Eiffel Tower", "url": "https://e.org/eiffel",
        "snippet": "The Eiffel Tower is a lattice tower on the Champ de Mars in Paris, France."}]


def test_what_it_learned_online_answers_the_same_question_offline(make_device, monkeypatch):
    """The complaint this exists to answer: pull the network and the device went back to "nothing on
    this device relates to that" about a thing it had read minutes earlier."""
    monkeypatch.setenv("EDGE_SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setattr(online, "search", lambda q, limit=5: list(HIT))
    d, _ = make_device("devA", "site1")

    assert d.ask("what is the Eiffel Tower")["sources"] == ["web"]

    d.outbox.kv_set("online", False)               # the network goes away
    out = d.ask("what is the Eiffel Tower")
    assert out["grounded"] is True and out["from_learned"] is True
    assert out["sources"] == ["device"], "it is answering from itself now, not from the internet"
    assert "Eiffel" in out["answer"]


def test_what_it_learned_answers_a_different_question_offline(make_device, monkeypatch):
    """Not a cache keyed on the question: the text is searchable like anything else it holds."""
    monkeypatch.setenv("EDGE_SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setattr(online, "search", lambda q, limit=5: list(HIT))
    d, _ = make_device("devA", "site1")
    d.ask("what is the Eiffel Tower")

    d.outbox.kv_set("online", False)
    out = d.ask("which tower is on the Champ de Mars")
    assert out["grounded"] is True and out["from_learned"] is True


def test_looking_the_same_page_up_twice_keeps_one_copy(make_device, monkeypatch):
    monkeypatch.setenv("EDGE_SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setattr(online, "search", lambda q, limit=5: list(HIT))
    d, _ = make_device("devA", "site1")
    d.ask("what is the Eiffel Tower")
    d.ask("tell me about the Eiffel Tower")
    assert d.store.count({"type": "memory", "kind": "learned"}) == 1


def test_a_cached_page_and_a_fresh_fetch_are_one_source(make_device, monkeypatch):
    monkeypatch.setenv("EDGE_SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setattr(online, "search", lambda q, limit=5: list(HIT))
    d, _ = make_device("devA", "site1")
    d.set_retrieval_mode("online")                 # always goes out, so both would otherwise be present
    d.ask("what is the Eiffel Tower")
    out = d.ask("what is the Eiffel Tower")
    assert [u["url"] for u in out["used"]] == ["https://e.org/eiffel"]


def test_what_it_learned_never_leaves_the_device(make_device, monkeypatch):
    """It is stored like a technician note: local, and never queued for the cloud."""
    monkeypatch.setenv("EDGE_SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setattr(online, "search", lambda q, limit=5: list(HIT))
    d, _ = make_device("devA", "site1")
    before = d.outbox.counts()
    d.ask("what is the Eiffel Tower")
    assert d.outbox.counts() == before
    learned = d.recall("Eiffel", limit=5, kind="learned")
    assert learned and all(m["share_state"] == "local" for m in learned)


def test_a_question_about_this_device_never_leaves_it(make_device, monkeypatch):
    """An overview question is about this machine's own record. There is no answer to it on the
    internet, and sending it out would leak what the device is for."""
    called = []
    monkeypatch.setenv("EDGE_SEARCH_PROVIDER", "duckduckgo")
    monkeypatch.setattr(online, "search", lambda q, limit=5: called.append(q) or [])
    d, _ = make_device("devA", "site1")
    d.set_retrieval_mode("online")

    out = d.ask("what problems have you seen recently?")
    assert out["mode"] == "overview"
    assert called == [], "a question about this device must never be sent to a search engine"
