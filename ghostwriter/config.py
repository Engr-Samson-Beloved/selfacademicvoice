import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.environ.get("GHOSTWRITER_PROVIDER", "gemini")
GROQ_API_KEY = "GROQ_API_KEY"
GEMINI_API_KEY = "GEMINI_API_KEY"
GEMINI_API_KEY2 = "GEMINI_API_KEY2"
GEMINI_API_SECRET = "GEMINI_API_SECRET"
# Groq retires models without notice: the previously hardcoded
# "llama-3.3-70b-versatile" now returns 404, which silently broke the documented
# Groq fallback for anyone who set a key. Overridable, like GEMINI_MODEL.
LLM_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")
KEEP_RATIO = float(os.environ.get("GHOSTWRITER_KEEP_RATIO", "0"))
REWRITE_CHUNK_SIZE = int(os.environ.get("GHOSTWRITER_CHUNK_SIZE", "8"))
REWRITE_MAX_WORKERS = int(os.environ.get("GHOSTWRITER_MAX_WORKERS", "4"))
REWRITE_TEMPERATURE = float(os.environ.get("GHOSTWRITER_TEMPERATURE", "0.8"))
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STYLE_PROMPT_FILE = Path("data/style_prompt.txt")
# Machine-readable voice profile written by `python -m ghostwriter.voiceprofile`.
# If the file is absent the voice gate is skipped and rewriting is unchanged.
VOICE_PROFILE_FILE = Path(os.environ.get("GHOSTWRITER_VOICE_PROFILE", "data/voice_profile.json"))
VOICE_GATE_ENABLED = os.environ.get("GHOSTWRITER_VOICE_GATE", "1") not in ("0", "false", "False")
VOICE_GATE_ATTEMPTS = int(os.environ.get("GHOSTWRITER_VOICE_GATE_ATTEMPTS", "2"))
INPUT_FILE = Path("data/input/report.txt")
OUTPUT_FILE = Path("data/rewritten/report_rewritten.txt")
VECTOR_DB_DIR = Path("data/vector_db")
COLLECTION_NAME = "ghostwriter"