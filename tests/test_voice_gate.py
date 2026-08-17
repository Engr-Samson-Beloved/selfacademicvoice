"""Tests for the measured voice profile and the rewrite-time voice gate.

Runs without API keys: ``llm.ask`` is replaced with a stub, so no Gemini or Groq
call is made and no quota is consumed.

    python tests/test_voice_gate.py     # or: pytest tests/test_voice_gate.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghostwriter import config, llm, rewrite, voiceprofile  # noqa: E402

PROFILE = voiceprofile.load_json(Path("data/voice_profile.json"))


# --------------------------------------------------------------- unit: rules

def test_banned_connective_flagged():
    orig = "Public libraries help the community."
    bad = "However, public libraries assist the community in many ways."
    problems = voiceprofile.sentence_violations(PROFILE, bad, orig)
    assert any("However" in p for p in problems), problems


def test_attested_connector_not_flagged():
    # "Furthermore" IS in the corpus, so it must survive the gate.
    orig = "Public libraries help the community."
    ok = "Furthermore, public libraries assist the community."
    problems = voiceprofile.sentence_violations(PROFILE, ok, orig)
    assert not any("Furthermore" in p for p in problems), problems


def test_semicolon_and_dash_flagged():
    orig = "The library opened in 1956."
    assert voiceprofile.sentence_violations(
        PROFILE, "The library opened in 1956; it moved later.", orig)
    assert voiceprofile.sentence_violations(
        PROFILE, "The library opened in 1956 — it moved later.", orig)


def test_length_inflation_flagged():
    orig = "Education depends on the right information."
    inflated = " ".join(["word"] * 80)
    problems = voiceprofile.sentence_violations(PROFILE, inflated, orig)
    assert any("shorten" in p for p in problems), problems


def test_comma_inflation_flagged():
    orig = "The library serves students."
    bad = ("The library, which is large, serves students, teachers, scholars, "
           "officials, and dropouts, among others.")
    problems = voiceprofile.sentence_violations(PROFILE, bad, orig)
    assert any("commas" in p for p in problems), problems


def test_clean_sentence_passes():
    orig = "Education solely depends on the availability of information."
    good = "Education depends only on having the right information at the right time."
    assert voiceprofile.sentence_violations(PROFILE, good, orig) == []


def test_profile_json_roundtrip(tmp_path=Path("data")):
    reloaded = voiceprofile.load_json(tmp_path / "voice_profile.json")
    assert reloaded.median_len == PROFILE.median_len
    assert reloaded.connectors and isinstance(reloaded.connectors[0], tuple)


# ------------------------------------------------------- integration: the gate

class FakeLLM:
    """Stands in for llm.ask. Replies to whatever numbers the prompt asks for."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = 0

    def __call__(self, prompt, system_prompt=None, temperature=None):
        self.calls += 1
        wanted = [int(m) for m in re.findall(r"^(\d+)\.", prompt, re.MULTILINE)]
        return "\n".join(
            f"{n}. {self.replies[n]}" for n in wanted if n in self.replies
        )


def _run_gate(rewrites, original_map, replies):
    fake = FakeLLM(replies)
    original_ask = llm.ask
    llm.ask = fake
    try:
        rewrite._apply_voice_gate(rewrites, original_map, "system")
    finally:
        llm.ask = original_ask
    return fake


def test_gate_repairs_a_breach():
    original_map = {1: "Public libraries help the community."}
    rewrites = {1: "However, public libraries assist the community greatly."}
    fake = _run_gate(rewrites, original_map,
                     {1: "Public libraries give real assistance to the community."})
    assert fake.calls == 1
    assert "However" not in rewrites[1]


def test_gate_rejects_a_non_improvement():
    """A 'fix' that still breaches must be discarded, not written back."""
    original_map = {1: "Public libraries help the community."}
    breaching = "However, public libraries assist the community greatly."
    rewrites = {1: breaching}
    _run_gate(rewrites, original_map,
              {1: "Moreover, public libraries assist the community greatly."})
    assert rewrites[1] == breaching, "a still-breaching repair was accepted"


def test_gate_rejects_repair_that_copies_the_original():
    """The voice gate must never undo the similarity gate."""
    orig = "Public libraries are established to provide timely information resources."
    rewrites = {1: "However, public libraries exist to supply prompt information."}
    _run_gate(rewrites, original_map={1: orig}, replies={1: orig})
    assert rewrites[1] != orig, "gate reintroduced a near-copy of the source"


def test_gate_is_noop_without_a_profile():
    """No profile file -> pipeline behaves exactly as before the gate existed."""
    saved = (rewrite._profile_cache, rewrite._profile_loaded, config.VOICE_PROFILE_FILE)
    rewrite._profile_cache, rewrite._profile_loaded = None, False
    config.VOICE_PROFILE_FILE = Path("data/does_not_exist.json")
    try:
        rewrites = {1: "However, this clearly breaches the voice rules."}
        fake = _run_gate(rewrites, {1: "Short original."}, {1: "irrelevant"})
        assert fake.calls == 0
        assert rewrites[1] == "However, this clearly breaches the voice rules."
    finally:
        rewrite._profile_cache, rewrite._profile_loaded, config.VOICE_PROFILE_FILE = saved


def test_gate_converges_and_stops():
    """When no repair is accepted the gate must stop, not spend every attempt."""
    original_map = {1: "Public libraries help the community."}
    rewrites = {1: "However, public libraries assist the community greatly."}
    fake = _run_gate(rewrites, original_map,
                     {1: "Moreover, public libraries assist the community greatly."})
    assert fake.calls == 1, f"expected one pass then stop, made {fake.calls}"


def test_full_rewrite_document_runs_with_gate():
    document = (
        "Public libraries are established to provide timely information resources "
        "to every member of the community. Education depends on the right "
        "information at the right time. The study shows that most users visit the "
        "library regularly."
    )
    replies = {
        1: "Public libraries exist so that prompt information resources reach all community members.",
        2: "Learning rests upon having correct information exactly when it is needed.",
        3: "Findings show most readers come to the library on a regular basis.",
    }

    class AnyNumberLLM(FakeLLM):
        def __call__(self, prompt, system_prompt=None, temperature=None):
            self.calls += 1
            wanted = [int(m) for m in re.findall(r"^(\d+)\.", prompt, re.MULTILINE)]
            return "\n".join(
                f"{n}. {self.replies.get(n, 'A short rewritten line for this item.')}"
                for n in wanted
            )

    fake = AnyNumberLLM(replies)
    original_ask = llm.ask
    llm.ask = fake
    try:
        out = rewrite.rewrite_document(document, "system")
    finally:
        llm.ask = original_ask

    assert out.strip(), "rewrite produced no output"
    assert fake.calls >= 1
    assert ";" not in out


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
