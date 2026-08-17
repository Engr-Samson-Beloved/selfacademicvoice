"""GhostWriter package.

Submodules are imported lazily (PEP 562) so offline tools can be used without
the full server dependency tree. ``ghostwriter.voiceprofile`` needs only pypdf;
importing it should not pull in fastapi, pydantic, chromadb or the LLM clients.

Attribute access still works exactly as before: ``ghostwriter.rewrite`` imports
``ghostwriter.rewrite`` on first use.
"""

import importlib

__all__ = [
    "config",
    "llm",
    "style",
    "rag",
    "rewrite",
    "models",
    "parse",
    "voiceprofile",
]


def __getattr__(name):
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
