"""The M.4 offline checklist with every connection attempt blocked and recorded (bench/offline_check.py), using the
lexical test embedder so it runs without model files. The full run with bge-small + the local LLM is the bench."""
from bench.offline_check import run
from tests.conftest import needs_cwru

pytestmark = needs_cwru


def test_every_device_function_works_with_zero_connection_attempts():
    res = run(real_models=False)
    failed = [c for c in res["checks"] if not c["ok"]]
    assert failed == [], failed
    assert res["connection_attempts_total"] == 0
    assert res["guard_self_test_caught_a_deliberate_request"] is True
