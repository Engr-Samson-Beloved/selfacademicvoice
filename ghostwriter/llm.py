import logging
import os
import re
import threading
import time

from . import config

log = logging.getLogger(__name__)

# Ceiling on how long to honour a provider's retry-after. Token-per-minute
# limits routinely ask for 20-30s, which the previous 10s cap could never satisfy.
MAX_RATE_LIMIT_WAIT = 60.0

_groq_client = None
_gemini_clients: dict[str, object] = {}

_client_lock = threading.Lock()


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        with _client_lock:
            if _groq_client is None:
                from groq import Groq

                _groq_client = Groq(api_key=os.getenv(config.GROQ_API_KEY))
    return _groq_client


def get_client():
    keys = _api_keys()
    if config.LLM_PROVIDER == "gemini" and keys:
        return _get_gemini_client(keys[0])
    return _get_groq_client()


def _api_keys():
    keys = []
    for name in (config.GEMINI_API_SECRET, config.GEMINI_API_KEY2):
        value = os.getenv(name, "").strip()
        if value:
            keys.append(value)
    return keys


def _get_gemini_client(api_key: str):
    if api_key not in _gemini_clients:
        with _client_lock:
            if api_key not in _gemini_clients:
                from google import genai

                _gemini_clients[api_key] = genai.Client(api_key=api_key)
    return _gemini_clients[api_key]


def _retry_delay(message: str) -> float:
    m = re.search(r"(?:retry|try again) in ([\d.]+)s", message, re.IGNORECASE)
    return float(m.group(1)) if m else 15.0


def _msg(e: Exception) -> str:
    """Exception text including its cause chain.

    _ask_gemini wraps the rotation's failures in a RuntimeError summary, so the
    original provider codes live on __cause__. Matching only str(e) would miss
    them and misclassify a quota failure as unrecoverable.
    """
    parts, seen, cur = [], set(), e
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(str(cur))
        cur = cur.__cause__ or cur.__context__
    return " | ".join(parts)


def _is_quota_error(e: Exception) -> bool:
    msg = _msg(e)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


def _is_retryable_quota(e: Exception) -> bool:
    msg = _msg(e)
    return "429" in msg and "perminute" in msg.lower()


def _is_connection_error(e: Exception) -> bool:
    name = type(e).__name__
    return any(
        t in name
        for t in ("Connect", "Timeout", "Protocol", "RemoteProtocol", "Network", "Unavailable")
    ) or "disconnected" in str(e).lower()


def _is_model_unavailable(e: Exception) -> bool:
    msg = _msg(e)
    return "404" in msg or "NOT_FOUND" in msg


def _is_overloaded(e: Exception) -> bool:
    msg = _msg(e)
    return "503" in msg or "UNAVAILABLE" in msg


def _is_server_error(e: Exception) -> bool:
    msg = _msg(e)
    return "500" in msg or "INTERNAL" in msg


def _is_fallback_error(e: Exception) -> bool:
    """Errors that should hand the request to the fallback provider.

    Quota exhaustion belongs here and was missing: the README advertises
    "automatic fallback to Groq when the Gemini quota is exhausted", but 429 /
    RESOURCE_EXHAUSTED was not in this set, so a quota-exhausted job failed
    outright even with GROQ_API_KEY configured. _is_quota_error existed for this
    and was never called.

    By the time an exception reaches here the per-model and per-key rotation in
    _ask_gemini has already been exhausted, including the sleep-and-retry for
    per-minute limits, so there is nothing left to wait for on Gemini's side.
    """
    return (
        _is_model_unavailable(e)
        or _is_server_error(e)
        or _is_quota_error(e)
        or _is_overloaded(e)
    )


def _groq_available() -> bool:
    return bool(os.getenv(config.GROQ_API_KEY))


def _gemini_generate(api_key: str, model: str, prompt: str, system_prompt: str, temperature: float) -> str:
    from google.genai import types

    client = _get_gemini_client(api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
        ),
    )
    text = (response.text or "").strip()
    if not text:
        # Same failure mode as Groq: an empty completion parses to zero
        # sentences and silently restores the source text for the whole chunk,
        # so it must raise and let the retry/rotation handle it.
        raise RuntimeError(f"{model} returned an empty completion")
    return text


# Floor on the time any one model gets, so a slow first model cannot leave the
# rest of the rotation unattempted.
MIN_MODEL_BUDGET = 25.0

DEFAULT_GEMINI_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
]


def _available_models():
    raw = os.environ.get("GEMINI_MODELS", "")
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return [config.GEMINI_MODEL] + [m for m in DEFAULT_GEMINI_MODELS if m != config.GEMINI_MODEL]


def _classify(e: Exception) -> str:
    msg = str(e)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "daily quota exhausted" if "PerDay" in msg else "rate limited"
    if "503" in msg or "UNAVAILABLE" in msg:
        return "overloaded"
    if "500" in msg or "INTERNAL" in msg:
        return "server error"
    if "404" in msg or "NOT_FOUND" in msg:
        return "not available to this key"
    if "API_KEY_INVALID" in msg or "API key not valid" in msg:
        return "invalid key"
    return type(e).__name__


def _ask_gemini(prompt: str, system_prompt: str, temperature: float) -> str:
    keys = _api_keys()
    if not keys:
        raise RuntimeError("No GEMINI_API_KEY set")
    last_err = None
    outcomes: list[str] = []
    models = _available_models()
    deadline = time.monotonic() + 150.0
    for index, model in enumerate(models):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            outcomes.append(f"{model}: not tried (deadline)")
            continue
        # Share the remaining budget with the models still to come. Without
        # this, one model timing out on connect consumed the whole window and
        # every healthy fallback was skipped with "not tried (deadline)".
        left = len(models) - index
        slice_deadline = time.monotonic() + max(MIN_MODEL_BUDGET, remaining / left)
        model_deadline = min(deadline, slice_deadline)
        for api_key in keys:
            try:
                return _ask_gemini_with_key(
                    api_key, model, prompt, system_prompt, temperature, model_deadline
                )
            except Exception as e:
                last_err = e
                outcomes.append(f"{model}: {_classify(e)}")

    if last_err is not None:
        # Previously this re-raised only the LAST model's error, so a request
        # that failed on overload across the whole rotation surfaced as whatever
        # the final fallback model happened to say - usually a quota message
        # naming a model that was never the real problem.
        summary = "; ".join(outcomes)
        raise RuntimeError(f"all Gemini models failed -> {summary}") from last_err
    raise RuntimeError("No Gemini key/model available")


def _ask_gemini_with_key(
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    deadline: float | None = None,
) -> str:
    last_err = None
    attempts = 3
    budget = 55.0
    start = time.monotonic()

    def _remaining() -> float:
        left = budget - (time.monotonic() - start)
        if deadline is not None:
            left = min(left, deadline - time.monotonic())
        return left

    for attempt in range(attempts):
        if _remaining() <= 0:
            break
        try:
            return _gemini_generate(api_key, model, prompt, system_prompt, temperature)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                last_err = e
                if "quota" in msg.lower() and "billing" in msg.lower():
                    if _is_retryable_quota(e):
                        delay = min(_retry_delay(msg), 10.0)
                        if 0 < delay < _remaining():
                            time.sleep(delay)
                            continue
                    break
                delay = min(_retry_delay(msg), 10.0)
                if 0 < delay < _remaining():
                    time.sleep(delay)
                    continue
                break
            if "503" in msg or "UNAVAILABLE" in msg:
                last_err = e
                delay = min([2.0, 4.0, 8.0][attempt], 10.0)
                if 0 < delay < _remaining():
                    time.sleep(delay)
                    continue
                break
            if "500" in msg or "INTERNAL" in msg:
                last_err = e
                delay = min([2.0, 4.0, 8.0][attempt], 10.0)
                if 0 < delay < _remaining():
                    time.sleep(delay)
                    continue
                break
            if _is_connection_error(e):
                last_err = e
                delay = min([2.0, 4.0, 8.0][attempt], 10.0)
                if 0 < delay < _remaining():
                    time.sleep(delay)
                    continue
                break
            if "404" in msg or "NOT_FOUND" in msg:
                last_err = e
                break
            raise
    if last_err is not None:
        raise last_err
    raise RuntimeError("No Gemini key/model available")


def _ask_groq(prompt: str, system_prompt: str, temperature: float) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    last_err = None
    for attempt in range(3):
        try:
            kwargs = {
                "model": config.LLM_MODEL,
                "messages": messages,
                "temperature": temperature,
                # Reasoning models spend the completion budget thinking and can
                # return empty content otherwise. Observed with gpt-oss-120b at
                # 16 sentences: 0 characters back, so every sentence in the chunk
                # silently fell back to the original text.
                "max_completion_tokens": config.GROQ_MAX_TOKENS,
            }
            if config.GROQ_REASONING_EFFORT:
                kwargs["reasoning_effort"] = config.GROQ_REASONING_EFFORT
            response = _get_groq_client().chat.completions.create(**kwargs)
            content = (response.choices[0].message.content or "").strip()
            if not content:
                # An empty completion is a failure, not an answer. Returning ""
                # here parses to zero sentences and silently restores the
                # source text for the whole chunk.
                raise RuntimeError(
                    f"{config.LLM_MODEL} returned an empty completion "
                    f"(finish_reason={getattr(response.choices[0], 'finish_reason', '?')}); "
                    "lower GHOSTWRITER_CHUNK_SIZE or raise GROQ_MAX_TOKENS"
                )
            return content
        except Exception as e:
            last_err = e
            msg = str(e)
            if "429" in msg or "RATE_LIMIT" in msg or "rate limit" in msg.lower():
                # Honour the server's own retry-after. Capping it at 10s meant a
                # provider asking for 21.5s was retried at 10s and failed again
                # every time - the cap guaranteed the retry could not succeed.
                # The small buffer avoids landing exactly on the boundary.
                delay = min(_retry_delay(msg) + 1.0, MAX_RATE_LIMIT_WAIT)
                if attempt < 2 and delay > 0:
                    log.info("rate limited by %s; waiting %.0fs", config.LLM_MODEL, delay)
                    time.sleep(delay)
                    continue
            raise
    raise last_err


def ask(prompt: str, system_prompt: str = None, temperature: float = 0.4) -> str:
    if config.LLM_PROVIDER == "gemini":
        try:
            return _ask_gemini(prompt, system_prompt, temperature)
        except Exception as e:
            if not _is_fallback_error(e):
                raise
            if _groq_available():
                try:
                    return _ask_groq(prompt, system_prompt, temperature)
                except Exception as groq_err:
                    raise RuntimeError(
                        f"Gemini API unavailable and Groq fallback failed: {groq_err}"
                    ) from e
            if _is_quota_error(e):
                raise RuntimeError(
                    "Gemini quota exhausted and no GROQ_API_KEY is set for automatic "
                    "fallback. Free-tier keys allow only a small number of requests "
                    "per day per model; either wait for the quota to reset, enable "
                    "billing on the Google project, or set GROQ_API_KEY."
                ) from e
            raise RuntimeError(
                "Gemini API unavailable and no GROQ_API_KEY is set for automatic fallback."
            ) from e
    return _ask_groq(prompt, system_prompt, temperature)
