# GhostWriter

AI ghostwriting assistant that rewrites documents in a specific author's voice. It rewrites individual sentences with a similarity gate so the output keeps your meaning but avoids verbatim AI-sounding text.

## Features

- Rewrites pasted text or uploaded files (`.docx`, `.pdf`) in the author's voice
- Sentence-level rewriting with a similarity check: sentences too close to the original are regenerated
- **Measured voice profile**: the style prompt is generated from the author's own corpus, not hand-written (`ghostwriter/voiceprofile.py`)
- **Voice gate**: after rewriting, sentences breaching the author's measured habits are re-prompted with the specific breach named
- Keeps headings, paragraphs and document structure intact for `.docx` output
- Rejoins sentences that were split across paragraphs (e.g. a continuation paragraph starting lowercase after one that doesn't end with punctuation)
- RAG retrieval over an embedded document library (ChromaDB + sentence-transformers)
- Model rotation with retry/fallback across Gemini and Groq
- Automatic fallback to Groq when the Gemini quota is exhausted (if `GROQ_API_KEY` is set)
- FastAPI service with auto-generated Swagger docs at `/docs`

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
GHOSTWRITER_CHUNK_SIZE=40
```

The style prompt lives in `data/style_prompt.txt` and the RAG vector DB in `data/vector_db/`. If the DB is missing, build it from `data/dataset/`:

```bash
python scripts/embed.py
```

## Voice profile

`data/style_prompt.txt` and `data/voice_profile.json` are **generated**, not
hand-edited. Both are derived from a corpus of the author's own writing:

```bash
python -m ghostwriter.voiceprofile <corpus> -o data/style_prompt.txt
```

`<corpus>` is a `.pdf`/`.txt` file or a directory of them. This writes two files:

- `data/style_prompt.txt` — the system prompt, with every figure measured from the corpus
- `data/voice_profile.json` — the machine-readable profile the voice gate reads at request time

Measured traits include the sentence-length distribution and deciles, comma and
conjunction rates, the connector inventory **plus an explicit never-use list built
from what is absent from the corpus**, citation forms, mid-sentence capitalisation,
and the lexicon split into transferable phrasing versus topic-bound vocabulary that
is gated off for unrelated subjects.

To audit any output against the profile:

```bash
python -m ghostwriter.voiceprofile <corpus> --check output.docx
```

This prints a per-metric comparison and exits non-zero if anything drifted, so it
can be used in CI.

### How the voice gate works

`rewrite_document()` runs two gates in sequence:

1. **Similarity gate** (pre-existing) — measures token overlap against the *source*
   sentence and regenerates near-copies.
2. **Voice gate** — measures each rewrite against the *author's profile* and
   re-prompts breaches, naming each one ("remove \"However\" — this author never
   uses it", "cut to at most 4 commas").

The voice gate is deliberately limited to per-sentence, deterministic breaches:
banned connectives, forbidden punctuation, and length or comma inflation relative
to the original. Distributional traits (median length, spread) cannot be enforced
there, because correcting spread means merging or splitting sentences and the
pipeline requires exactly one rewrite per source sentence for reassembly. Those
traits are reported instead, via a `voice drift:` log line after each document.

A repair is accepted only if it **reduces** the breach count and stays below the
similarity threshold, so the voice gate can never undo the similarity gate. If
`data/voice_profile.json` is absent the gate is skipped and rewriting behaves
exactly as it did before.

## Tests

```bash
python tests/test_voice_gate.py     # or: pytest tests/test_voice_gate.py
```

`llm.ask` is stubbed, so the suite needs no API keys and consumes no quota.

## Run

```bash
uvicorn api.main:app --reload --port 8000
```

- Web UI: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## API

### `POST /rewrite`

Submits a rewrite **job** and returns `202` with a `job_id` immediately. The rewrite
runs in the background, so the request never hangs past Cloudflare-style gateway timeouts.
Poll `GET /rewrite/{job_id}` until `status` is `done` or `error`, then download the result.

Accepts `multipart/form-data`. Provide exactly one of `draft` or `file`.

| Field | Type | Required | Description |
|---|---|---|---|
| `draft` | string | if no `file` | Document text to rewrite |
| `file` | file | if no `draft` | A `.docx` or `.pdf` file to rewrite |
| `system_prompt_override` | string | no | Replaces the default author-style prompt |

- **`POST /rewrite`** → `202 { "job_id": "...", "status": "running" }`
- **`GET /rewrite/{job_id}`** → `{ "status": "pending" | "running" | "done" | "error", ... }`
  - `draft` jobs: when `done`, includes `{ "rewritten": "..." }`
  - `file` jobs: when `done`, includes `{ "download_url": "/rewrite/{job_id}/download" }`
- **`GET /rewrite/{job_id}/download`** → the rewritten `.docx` (attachment named `<original>_rewritten.docx`)

Examples:

```bash
# submit a text job
curl -X POST http://localhost:8000/rewrite \
  -F 'draft=The quick brown fox jumps over the lazy dog.'
# -> 202 {"job_id":"abc123","status":"running"}
curl http://localhost:8000/rewrite/abc123
# -> {"job_id":"abc123","status":"done","rewritten":"..."}

# submit a file job
curl -X POST http://localhost:8000/rewrite -F 'file=@draft.docx'
# -> 202 {"job_id":"def456","status":"running"}
curl http://localhost:8000/rewrite/def456
# -> {"job_id":"def456","status":"done","download_url":"/rewrite/def456/download"}
curl -O http://localhost:8000/rewrite/def456/download
```

Errors return JSON with a `detail` field: `400` for missing/empty input or unsupported file types, `404` for unknown/expired jobs, `500` if the rewrite pipeline fails.

## Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Gemini API key |
| `GROQ_API_KEY` | — | Groq API key (fallback provider) |
| `GHOSTWRITER_PROVIDER` | `gemini` | Primary LLM provider |
| `GEMINI_MODEL` | `gemini-3.7-flash` | Primary Gemini model; others are tried in rotation on failure |
| `GHOSTWRITER_CHUNK_SIZE` | `8` | Sentences per rewrite request |
| `GHOSTWRITER_MAX_WORKERS` | `4` | Chunks rewritten in parallel (one LLM call per worker) |
| `GHOSTWRITER_KEEP_RATIO` | `0` | Fraction of sentences kept verbatim. `0` rewrites every sentence |
| `GHOSTWRITER_TEMPERATURE` | `0.8` | Sampling temperature |
| `GHOSTWRITER_VOICE_PROFILE` | `data/voice_profile.json` | Measured profile the voice gate reads |
| `GHOSTWRITER_VOICE_GATE` | `1` | Set `0` to disable the voice gate |
| `GHOSTWRITER_VOICE_GATE_ATTEMPTS` | `2` | Max voice-repair passes per document |

> **Gemini key names.** `_api_keys()` in `ghostwriter/llm.py` reads
> `GEMINI_API_SECRET` and `GEMINI_API_KEY2`, in that order — **not**
> `GEMINI_API_KEY`. A key set only as `GEMINI_API_KEY` is silently ignored.

## Performance / timeouts

Chunks are rewritten in parallel (`GHOSTWRITER_MAX_WORKERS`, default 4), so wall-clock time is roughly `chunks / workers × per-call latency` rather than `chunks × per-call latency`. A single synchronous request still must finish inside your proxy's timeout (e.g. Cloudflare's ~100s): for long documents set `GHOSTWRITER_CHUNK_SIZE` (e.g. `40`) and `GHOSTWRITER_MAX_WORKERS` (e.g. `4`) so the request completes well under that limit.

## Project layout

```
api/               FastAPI app
ghostwriter/       core rewrite engine (config, llm, rag, parse, style, voiceprofile)
scripts/           data pipeline: extract, split, clean, embed
tests/             voice profile and voice gate tests (no API keys needed)
data/              style prompt, voice profile, vector DB, inputs and rewritten output
```

Note `*.pdf` is gitignored, so an author corpus kept in the repo is not tracked.
`data/style_prompt.txt` is generated from it — keep the corpus somewhere durable or
the prompt cannot be regenerated.

## Credits

The voice profile is measured from the academic writing of **Sulieman Sambo** and
**Abdu Azarema** (Career Point University, Kota):

> Sambo, S. and Azarema, A. (2017). *Public Libraries as Tools for Socio-Economic
> and Political Development: A Case Study of Selected Public Libraries in Kota
> Region, Rajasthan.* Paper presented at the National Conference on Development and
> Governance: Issues & Prospects, Department of Public Administration,
> S. S. Jain Subodh PG (Autonomous) College, Jaipur, 27–28 January 2017.

Short excerpts from that paper appear in `data/style_prompt.txt` as style
exemplars, and all figures in the generated profile are measured from it. The
source PDF itself is not distributed with this repository.

## Deployment

Containerized deployment (Docker / Railway) docs coming soon. Note `data/vector_db` and `data/dataset/` are gitignored — rebuild or restore them on the server.
