import asyncio
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

import ghostwriter

app = FastAPI(
    title="GhostWriter API",
    version="1.1.0",
    openapi_tags=[
        {
            "name": "rewrite",
            "description": "Rewrite documents in the author's voice.",
        },
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_JOB_TTL = 1800.0
_jobs: dict[str, dict] = {}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


def _new_job(kind: str, filename: str | None = None) -> dict:
    job_id = uuid.uuid4().hex[:16]
    job = {
        "id": job_id,
        "kind": kind,
        "status": "pending",
        "error": None,
        "result": None,
        "filename": filename,
        "out_name": None,
        "created": time.time(),
    }
    _jobs[job_id] = job
    return job


def _cleanup_jobs():
    now = time.time()
    for jid in [j for j, jb in list(_jobs.items()) if now - jb["created"] > _JOB_TTL]:
        _jobs.pop(jid, None)


def _do_text_rewrite(draft: str, system_prompt: str) -> str:
    result = ghostwriter.rewrite.rewrite_document(draft, system_prompt)

    output_dir = Path("data/rewritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"rewrite_{ts}.txt"
    output_path.write_text(result, encoding="utf8")

    return result


def _do_file_rewrite(file_bytes: bytes, filename: str, system_prompt: str) -> bytes:
    suffix = Path(filename).suffix.lower()

    if suffix == ".docx":
        result_bytes = ghostwriter.rewrite.rewrite_docx(file_bytes, system_prompt)
    else:
        text = ghostwriter.parse.parse_pdf(file_bytes)
        result = ghostwriter.rewrite.rewrite_document(text, system_prompt)
        result_bytes = ghostwriter.parse._make_docx_from_text(result)

    output_dir = Path("data/rewritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = Path(filename).stem + "_rewritten.docx"
    output_path = output_dir / f"rewrite_{ts}_{out_name}"
    output_path.write_bytes(result_bytes)

    return result_bytes


def _is_transient_error(e: Exception) -> bool:
    msg = str(e)
    name = type(e).__name__
    return any(
        t in msg
        for t in ("503", "UNAVAILABLE", "500", "INTERNAL", "429", "RESOURCE_EXHAUSTED", "quota", "rate limit", "RATE_LIMIT", "disconnected")
    ) or any(
        t in name
        for t in ("Connect", "Timeout", "Protocol", "RemoteProtocol", "Network")
    )


def _friendly_error(e: Exception) -> str:
    msg = str(e).strip()
    if not msg:
        return f"{type(e).__name__}"
    return f"{type(e).__name__}: {msg}"


async def _run_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        return
    job["status"] = "running"

    def _run_once():
        if job["kind"] == "text":
            return asyncio.to_thread(
                _do_text_rewrite, job["draft"], job["system_prompt"]
            )
        return asyncio.to_thread(
            _do_file_rewrite, job["file_bytes"], job["filename"], job["system_prompt"]
        )

    backoffs = [30, 60]
    for attempt in range(3):
        try:
            job["result"] = await _run_once()
            job["status"] = "done"
            return
        except Exception as e:
            if not _is_transient_error(e) or attempt >= len(backoffs):
                job["status"] = "error"
                job["error"] = _friendly_error(e)
                return
            time.sleep(backoffs[attempt])

    job["status"] = "error"
    job["error"] = (
        "The rewrite could not be completed because Gemini was temporarily "
        "overloaded or rate-limited after several attempts. Please wait a few "
        "minutes and try again."
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GhostWriter</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; margin: 0; background: #f5f6fa; color: #222; }
  .wrap { max-width: 760px; margin: 0 auto; padding: 32px 20px; }
  h1 { font-size: 24px; margin: 0 0 4px; }
  .sub { color: #666; margin-bottom: 24px; }
  .tabs { display: flex; gap: 8px; margin-bottom: 16px; }
  .tab { padding: 8px 18px; border: 1px solid #ccc; background: #fff; border-radius: 8px; cursor: pointer; font-size: 14px; }
  .tab.active { background: #2563eb; color: #fff; border-color: #2563eb; }
  .panel { display: none; background: #fff; border: 1px solid #e2e4ea; border-radius: 12px; padding: 20px; }
  .panel.active { display: block; }
  .drop { border: 2px dashed #bbb; border-radius: 10px; padding: 28px; text-align: center; color: #555; cursor: pointer; }
  .drop.drag { border-color: #2563eb; background: #eff6ff; }
  label { display: block; font-weight: 600; font-size: 14px; margin: 14px 0 6px; }
  textarea { width: 100%; min-height: 180px; padding: 10px; border: 1px solid #ccc; border-radius: 8px; font: 14px/1.5 inherit; resize: vertical; }
  button.rewrite { margin-top: 16px; padding: 10px 24px; background: #2563eb; color: #fff; border: none; border-radius: 8px; font-size: 15px; cursor: pointer; }
  button.rewrite:disabled { background: #9ca3af; cursor: not-allowed; }
  button.copy { margin-top: 8px; padding: 6px 14px; background: #eee; border: 1px solid #ccc; border-radius: 6px; font-size: 13px; cursor: pointer; }
  .status { margin-top: 14px; font-size: 14px; min-height: 20px; }
  .status.ok { color: #16a34a; }
  .status.err { color: #dc2626; }
  .output { margin-top: 16px; }
  .out-label { font-weight: 600; font-size: 14px; margin-bottom: 6px; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #2563eb; border-top-color: transparent; border-radius: 50%; animation: spin .7s linear infinite; vertical-align: -2px; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="wrap">
  <h1>GhostWriter</h1>
  <p class="sub">Rewrite documents in the author's voice</p>

  <div class="tabs">
    <button class="tab active" data-tab="file">Upload File</button>
    <button class="tab" data-tab="text">Paste Text</button>
  </div>

  <div class="panel active" id="panel-file">
    <div class="drop" id="drop">
      <p><strong>Drag &amp; drop</strong> a .docx or .pdf here, or click to browse</p>
      <input type="file" id="fileInput" accept=".docx,.pdf" style="display:none">
    </div>
    <div id="fileMeta" style="margin-top:10px;font-size:14px;color:#555"></div>
    <button class="rewrite" id="fileBtn">Rewrite</button>
    <div class="status" id="fileStatus"></div>
  </div>

  <div class="panel" id="panel-text">
    <label for="textInput">Paste your draft below</label>
    <textarea id="textInput" placeholder="Paste the document text you want rewritten..."></textarea>
    <button class="rewrite" id="textBtn">Rewrite</button>
    <div class="status" id="textStatus"></div>
    <div class="output" id="textOutput" style="display:none">
      <div class="out-label">Rewritten text</div>
      <textarea id="textResult" readonly></textarea>
      <button class="copy" id="copyBtn">Copy</button>
    </div>
  </div>
</div>

<script>
var tabs = document.querySelectorAll('.tab');
tabs.forEach(function(tab) {
  tab.addEventListener('click', function() {
    tabs.forEach(function(t) { t.classList.remove('active'); });
    document.querySelectorAll('.panel').forEach(function(p) { p.classList.remove('active'); });
    tab.classList.add('active');
    document.getElementById('panel-' + tab.dataset.tab).classList.add('active');
  });
});

function setStatus(el, msg, type) {
  el.className = 'status' + (type ? ' ' + type : '');
  el.innerHTML = msg;
}

function pollJob(jobId, kind) {
  return fetch('/rewrite/' + jobId).then(function(resp) {
    if (!resp.ok) { throw new Error('Status ' + resp.status + ': failed to check rewrite job'); }
    return resp.json();
  }).then(function(job) {
    if (job.status === 'error') { throw new Error(job.error || 'Rewrite failed'); }
    if (job.status === 'done') {
      if (kind === 'file') {
        return fetch(job.download_url).then(function(r) {
          if (!r.ok) { throw new Error('Status ' + r.status + ': download failed'); }
          var cd = r.headers.get('Content-Disposition') || '';
          var name = 'rewritten.docx';
          var m = cd.match(/filename="([^"]+)"/);
          if (m) name = m[1];
          return r.blob().then(function(blob) { return { blob: blob, name: name }; });
        });
      }
      return job.rewritten;
    }
    return new Promise(function(resolve) {
      setTimeout(function() { resolve(pollJob(jobId, kind)); }, 3000);
    });
  });
}

var drop = document.getElementById('drop');
var fileInput = document.getElementById('fileInput');
var fileMeta = document.getElementById('fileMeta');
var fileBtn = document.getElementById('fileBtn');
var fileStatus = document.getElementById('fileStatus');

drop.addEventListener('click', function() { fileInput.click(); });
drop.addEventListener('dragover', function(e) { e.preventDefault(); drop.classList.add('drag'); });
drop.addEventListener('dragleave', function() { drop.classList.remove('drag'); });
drop.addEventListener('drop', function(e) { e.preventDefault(); drop.classList.remove('drag'); fileInput.files = e.dataTransfer.files; showFile(); });
fileInput.addEventListener('change', showFile);

function showFile() {
  var f = fileInput.files[0];
  if (!f) { fileMeta.textContent = ''; return; }
  fileMeta.textContent = f.name + ' (' + (f.size / 1024).toFixed(1) + ' KB)';
}

fileBtn.addEventListener('click', function() {
  var f = fileInput.files[0];
  if (!f) { setStatus(fileStatus, 'Choose a file first', 'err'); return; }

  var fd = new FormData();
  fd.append('file', f);
  fileBtn.disabled = true;
  setStatus(fileStatus, '<span class="spinner"></span>Rewriting... (this may take a minute)');

  fetch('/rewrite', { method: 'POST', body: fd })
    .then(function(resp) {
      if (!resp.ok) {
        return resp.text().then(function(txt) {
          var detail = 'Rewrite failed';
          try { detail = JSON.parse(txt).detail || detail; }
          catch (e) { detail = txt.slice(0, 200) || detail; }
          throw new Error('Status ' + resp.status + ': ' + detail);
        });
      }
      return resp.json();
    })
    .then(function(job) { return pollJob(job.job_id, 'file'); })
    .then(function(r) {
      var url = URL.createObjectURL(r.blob);
      var a = document.createElement('a');
      a.href = url; a.download = r.name;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(function() { URL.revokeObjectURL(url); }, 4000);
      setStatus(fileStatus, 'Done — downloading ' + r.name, 'ok');
    })
    .catch(function(err) { setStatus(fileStatus, err.message, 'err'); })
    .finally(function() { fileBtn.disabled = false; });
});

var textBtn = document.getElementById('textBtn');
var textInput = document.getElementById('textInput');
var textStatus = document.getElementById('textStatus');
var textOutput = document.getElementById('textOutput');
var textResult = document.getElementById('textResult');

textBtn.addEventListener('click', function() {
  var draft = textInput.value.trim();
  if (!draft) { setStatus(textStatus, 'Paste some text first', 'err'); return; }

  var fd = new FormData();
  fd.append('draft', draft);
  textBtn.disabled = true;
  setStatus(textStatus, '<span class="spinner"></span>Rewriting... (this may take a minute)');
  textOutput.style.display = 'none';

  fetch('/rewrite', { method: 'POST', body: fd })
    .then(function(resp) {
      if (!resp.ok) {
        return resp.text().then(function(txt) {
          var detail = 'Rewrite failed';
          try { detail = JSON.parse(txt).detail || detail; }
          catch (e) { detail = txt.slice(0, 200) || detail; }
          throw new Error('Status ' + resp.status + ': ' + detail);
        });
      }
      return resp.json();
    })
    .then(function(job) { return pollJob(job.job_id, 'text'); })
    .then(function(rewritten) {
      textResult.value = rewritten;
      textOutput.style.display = 'block';
      setStatus(textStatus, 'Done', 'ok');
    })
    .catch(function(err) { setStatus(textStatus, err.message, 'err'); })
    .finally(function() { textBtn.disabled = false; });
});

document.getElementById('copyBtn').addEventListener('click', function() {
  navigator.clipboard.writeText(textResult.value);
});
</script>
</body>
</html>"""


@app.get("/health", include_in_schema=False)
def health():
    try:
        docs = ghostwriter.rag.get_all_documents()
        count = len(docs)
    except Exception:
        count = 0
    return {"status": "ok", "documents_in_db": count}


@app.post(
    "/rewrite",
    tags=["rewrite"],
    summary="Submit a rewrite job",
    description=(
        "Submits the provided draft or uploaded file for rewriting in the author's voice. "
        "Accepts `multipart/form-data`. Provide exactly one of `draft` or `file`.\n\n"
        "The rewrite runs in the background. Returns `202` with a `job_id`; poll "
        "`GET /rewrite/{job_id}` until `status` is `done` or `error`. For file jobs, "
        "download the result from the returned `download_url`.\n\n"
        "- **Text input** (`draft`): on completion, the poll response contains `rewritten`.\n"
        "- **File input** (`file`, .docx or .pdf): on completion, download the rewritten "
        "`.docx` via `download_url`.\n\n"
        "Optionally pass `system_prompt_override` to replace the default author-style prompt."
    ),
    responses={
        202: {
            "description": "Job accepted and running in the background.",
            "content": {"application/json": {"example": {"job_id": "abc123", "status": "running"}}},
        },
        400: {
            "description": "Bad request: missing/empty input, or unsupported file type.",
            "content": {"application/json": {"example": {"detail": "Provide either 'draft' text or a 'file' upload"}}},
        },
    },
)
async def rewrite(
    draft: str | None = Form(
        None,
        description="Document text to rewrite. Ignored if `file` is provided.",
        examples=["Write the draft text here..."],
    ),
    file: UploadFile | None = File(
        None,
        description="A .docx or .pdf file to rewrite. Ignored if `draft` is provided.",
    ),
    system_prompt_override: str | None = Form(
        None,
        description="Optional. Replaces the default author-style system prompt used for rewriting.",
    ),
):
    if not file and not draft:
        raise HTTPException(status_code=400, detail="Provide either 'draft' text or a 'file' upload")

    system_prompt = system_prompt_override or ghostwriter.style.load_prompt()
    _cleanup_jobs()

    if file:
        filename = file.filename or "upload.docx"
        suffix = Path(filename).suffix.lower()
        if suffix not in (".docx", ".pdf"):
            raise HTTPException(status_code=400, detail="Only .docx and .pdf files are supported")

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="File is empty")

        if suffix == ".docx":
            try:
                ghostwriter.parse.docx_stats(file_bytes)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Not a valid .docx file (is it an image or an HTML document renamed to .docx?)",
                )

        job = _new_job("file", filename=filename)
        job["file_bytes"] = file_bytes
        job["system_prompt"] = system_prompt
        job["out_name"] = Path(filename).stem + "_rewritten.docx"
    else:
        if not draft.strip():
            raise HTTPException(status_code=400, detail="draft must not be empty")
        job = _new_job("text")
        job["draft"] = draft
        job["system_prompt"] = system_prompt

    asyncio.create_task(_run_job(job["id"]))
    return JSONResponse(status_code=202, content={"job_id": job["id"], "status": "running"})


@app.get("/rewrite/{job_id}", tags=["rewrite"], summary="Check a rewrite job's status")
def rewrite_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    if job["status"] == "done":
        if job["kind"] == "text":
            return {"job_id": job_id, "status": "done", "rewritten": job["result"]}
        return {
            "job_id": job_id,
            "status": "done",
            "download_url": f"/rewrite/{job_id}/download",
        }
    if job["status"] == "error":
        return {"job_id": job_id, "status": "error", "error": job["error"]}
    return {"job_id": job_id, "status": job["status"]}


@app.get("/rewrite/{job_id}/download", tags=["rewrite"], summary="Download a finished file rewrite")
def rewrite_download(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    if job["status"] != "done" or job["kind"] != "file":
        raise HTTPException(status_code=404, detail="Result not ready")
    return Response(
        content=job["result"],
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{job["out_name"]}"'},
    )


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))