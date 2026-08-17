from pathlib import Path

from . import config


def load_prompt() -> str:
    path = Path(config.STYLE_PROMPT_FILE)
    with open(path, "r", encoding="utf8") as f:
        return f.read().strip()
