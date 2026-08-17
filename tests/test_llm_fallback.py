"""Provider-fallback behaviour. No network calls and no API keys required."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghostwriter import config, llm  # noqa: E402

QUOTA_ERROR = (
    "ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': "
    "'You exceeded your current quota', 'status': 'RESOURCE_EXHAUSTED'}} "
    "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
)


def test_quota_error_is_a_fallback_error():
    """Regression: 429 was not in _is_fallback_error, so a quota-exhausted job
    failed outright despite the README promising a Groq fallback."""
    assert llm._is_fallback_error(Exception(QUOTA_ERROR))


def test_overload_and_server_errors_are_fallback_errors():
    assert llm._is_fallback_error(Exception("503 UNAVAILABLE"))
    assert llm._is_fallback_error(Exception("500 INTERNAL"))
    assert llm._is_fallback_error(Exception("404 NOT_FOUND"))


def test_ordinary_error_is_not_a_fallback_error():
    assert not llm._is_fallback_error(Exception("400 INVALID_ARGUMENT: bad request"))


def _with_stubs(gemini_exc, groq_reply, groq_key):
    saved = (llm._ask_gemini, llm._ask_groq, os.environ.get(config.GROQ_API_KEY))
    calls = {"groq": 0}

    def fake_gemini(*a, **k):
        raise Exception(gemini_exc)

    def fake_groq(*a, **k):
        calls["groq"] += 1
        return groq_reply

    llm._ask_gemini, llm._ask_groq = fake_gemini, fake_groq
    if groq_key:
        os.environ[config.GROQ_API_KEY] = groq_key
    else:
        os.environ.pop(config.GROQ_API_KEY, None)
    return saved, calls


def _restore(saved):
    llm._ask_gemini, llm._ask_groq, key = saved
    if key is None:
        os.environ.pop(config.GROQ_API_KEY, None)
    else:
        os.environ[config.GROQ_API_KEY] = key


def test_quota_exhaustion_falls_back_to_groq():
    saved, calls = _with_stubs(QUOTA_ERROR, "rewritten by groq", "test-key")
    try:
        assert llm.ask("prompt") == "rewritten by groq"
        assert calls["groq"] == 1, "Groq was never called"
    finally:
        _restore(saved)


def test_quota_exhaustion_without_groq_key_explains_itself():
    saved, _ = _with_stubs(QUOTA_ERROR, "unused", None)
    try:
        try:
            llm.ask("prompt")
        except RuntimeError as e:
            msg = str(e)
            assert "quota" in msg.lower(), msg
            assert "GROQ_API_KEY" in msg, msg
            return
        raise AssertionError("no error raised on exhausted quota")
    finally:
        _restore(saved)


def test_non_fallback_error_is_not_sent_to_groq():
    saved, calls = _with_stubs("400 INVALID_ARGUMENT", "unused", "test-key")
    try:
        try:
            llm.ask("prompt")
        except Exception:
            pass
        assert calls["groq"] == 0, "a non-fallback error was sent to Groq"
    finally:
        _restore(saved)


if __name__ == "__main__":
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
