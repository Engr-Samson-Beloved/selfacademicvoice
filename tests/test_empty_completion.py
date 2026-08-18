"""An empty completion must fail loudly, not silently restore the source text.

gpt-oss-120b returned zero characters for a 16-sentence chunk. _ask_groq passed
that back as a successful answer, _parse_numbered found no sentences, and every
sentence in the chunk fell back to its original wording. Across one document
that silently left 240 sentences unrewritten while every other signal - word
count, label count, gate breaches - reported success.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghostwriter import config, llm, rewrite  # noqa: E402


def _fake_groq_client(content):
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message, finish_reason="length")
    response = types.SimpleNamespace(choices=[choice])
    completions = types.SimpleNamespace(create=lambda **kw: response)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))


def _run(content):
    saved = llm._get_groq_client
    llm._get_groq_client = lambda: _fake_groq_client(content)
    try:
        return llm._ask_groq("prompt", "system", 0.5)
    finally:
        llm._get_groq_client = saved


def test_empty_string_raises():
    for bad in ("", "   ", "\n\n"):
        try:
            _run(bad)
        except Exception as e:
            assert "empty completion" in str(e), str(e)
            continue
        raise AssertionError(f"empty content {bad!r} was returned as success")


def test_none_content_raises():
    try:
        _run(None)
    except Exception as e:
        assert "empty completion" in str(e), str(e)
        return
    raise AssertionError("None content was returned as success")


def test_real_content_passes_through():
    assert _run("1. A properly rewritten sentence.") == "1. A properly rewritten sentence."


def test_error_names_the_remedy():
    try:
        _run("")
    except Exception as e:
        msg = str(e)
        assert "CHUNK_SIZE" in msg or "MAX_TOKENS" in msg, msg
        return
    raise AssertionError("no error raised")


def test_empty_response_does_not_silently_keep_the_original():
    """End to end: if every call comes back empty the job must fail, not return
    the source document dressed up as a rewrite."""
    document = ("Blockchain technology has applications far beyond cryptocurrency "
                "and is now used across many different industries worldwide. "
                "Supply chains benefit from records that cannot be altered later.")

    def empty_ask(*a, **k):
        raise RuntimeError("model returned an empty completion")

    saved = llm.ask
    llm.ask = empty_ask
    try:
        try:
            rewrite.rewrite_document(document, "system")
        except Exception:
            return
        raise AssertionError("a fully-empty provider produced a 'successful' rewrite")
    finally:
        llm.ask = saved


def test_groq_sends_a_completion_budget():
    """Without max_completion_tokens a reasoning model spends its whole budget
    thinking and returns nothing."""
    captured = {}

    def create(**kw):
        captured.update(kw)
        message = types.SimpleNamespace(content="1. ok")
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason="stop")]
        )

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )
    saved = llm._get_groq_client
    llm._get_groq_client = lambda: client
    try:
        llm._ask_groq("prompt", "system", 0.5)
    finally:
        llm._get_groq_client = saved

    assert captured.get("max_completion_tokens") == config.GROQ_MAX_TOKENS
    assert captured["max_completion_tokens"] > 0


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
