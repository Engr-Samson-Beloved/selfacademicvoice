import os
import re
import threading
import time

from . import config

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


def _is_quota_error(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


def _is_retryable_quota(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg and "perminute" in msg.lower()


def _is_connection_error(e: Exception) -> bool:
    name = type(e).__name__
    return any(
        t in name
        for t in ("Connect", "Timeout", "Protocol", "RemoteProtocol", "Network", "Unavailable")
    ) or "disconnected" in str(e).lower()


def _is_model_unavailable(e: Exception) -> bool:
    msg = str(e)
    return "404" in msg or "NOT_FOUND" in msg


def _is_overloaded(e: Exception) -> bool:
    msg = str(e)
    return "503" in msg or "UNAVAILABLE" in msg


def _is_server_error(e: Exception) -> bool:
    msg = str(e)
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
    return response.text.strip()


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


def _ask_gemini(prompt: str, system_prompt: str, temperature: float) -> str:
    keys = _api_keys()
    if not keys:
        raise RuntimeError("No GEMINI_API_KEY set")
    last_err = None
    deadline = time.monotonic() + 150.0
    for model in _available_models():
        if time.monotonic() >= deadline:
            break
        for api_key in keys:
            try:
                return _ask_gemini_with_key(
                    api_key, model, prompt, system_prompt, temperature, deadline
                )
            except Exception as e:
                last_err = e
    if last_err is not None:
        raise last_err
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
            response = _get_groq_client().chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            msg = str(e)
            if "429" in msg or "RATE_LIMIT" in msg or "rate limit" in msg.lower():
                delay = min(_retry_delay(msg), 10.0)
                if attempt < 2 and delay <= 10.0:
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
