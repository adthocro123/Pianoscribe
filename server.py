"""FastAPI server for the piano transcription app.

Each transcription runs pipeline.py as a subprocess; progress is read back
from the job directory's status.json.
"""

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
JOBS_DIR = ROOT / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Piano Sheet Transcriber")
processes: dict[str, subprocess.Popen] = {}


class TranscribeRequest(BaseModel):
    url: str
    separate: bool = False
    instrument: str = "piano"


@app.post("/api/transcribe")
def transcribe(req: TranscribeRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "Missing URL")
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir()
    (job_dir / "status.json").write_text(json.dumps({"stage": "starting", "done": False, "error": None}))

    cmd = [sys.executable, str(ROOT / "pipeline.py"), url, str(job_dir)]
    if req.separate:
        cmd.append("--separate")
    if req.instrument == "guitar":
        cmd.extend(["--instrument", "guitar"])
    log = open(job_dir / "pipeline.log", "w")
    processes[job_id] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
    return {"job_id": job_id}


@app.get("/api/jobs")
def list_jobs():
    items = []
    for d in JOBS_DIR.iterdir():
        status_file = d / "status.json"
        if not d.is_dir() or not status_file.exists():
            continue
        try:
            s = json.loads(status_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        items.append({
            "job_id": d.name,
            "title": s.get("title"),
            "thumbnail": s.get("thumbnail"),
            "duration": s.get("duration"),
            "tempo": s.get("tempo"),
            "done": s.get("done", False),
            "error": s.get("error"),
            "instrument": s.get("instrument", "piano"),
            "created": status_file.stat().st_mtime,
        })
    items.sort(key=lambda x: x["created"], reverse=True)
    return items[:50]


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    path = (JOBS_DIR / job_id).resolve()
    if path.parent != JOBS_DIR.resolve() or not path.is_dir():
        raise HTTPException(404, "Unknown job")
    proc = processes.pop(job_id, None)
    if proc and proc.poll() is None:
        proc.terminate()
    shutil.rmtree(path)
    return {"ok": True}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job_dir = JOBS_DIR / job_id
    status_file = job_dir / "status.json"
    if not status_file.exists():
        raise HTTPException(404, "Unknown job")
    status = json.loads(status_file.read_text())

    # If the worker died without writing a proper error, surface that.
    proc = processes.get(job_id)
    if proc and proc.poll() not in (None, 0) and not status.get("done"):
        tail = ""
        log = job_dir / "pipeline.log"
        if log.exists():
            tail = log.read_text()[-500:]
        status.update(stage="error", done=True, error=f"Worker crashed. {tail}")
    status["job_id"] = job_id
    return status


@app.get("/files/{job_id}/{filename}")
def job_file(job_id: str, filename: str):
    path = (JOBS_DIR / job_id / filename).resolve()
    if not path.is_file() or JOBS_DIR.resolve() not in path.parents:
        raise HTTPException(404, "Not found")
    return FileResponse(path)


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
