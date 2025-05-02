import os
import uuid
import shutil
import zipfile
import subprocess
import asyncio
from pathlib import Path
from typing import Optional, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import redis
import json

# Toggle Redis tracking
USE_REDIS = True

# Redis config
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

try:
    rdb = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    rdb.ping()
except Exception as e:
    if USE_REDIS:
        raise RuntimeError(f"Could not connect to Redis: {e}")
    rdb = None

# App initialization
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# File paths
BASE_DIR = Path("/mnt")
UPLOAD_ROOT = BASE_DIR / "mriqc_upload"
OUTPUT_ROOT = BASE_DIR / "mriqc_output"
RESULT_ROOT = BASE_DIR / "mriqc_results"
os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(RESULT_ROOT, exist_ok=True)

jobs: Dict[str, Dict] = {}

@app.get("/health")
def health():
    return {"status": "ok", "redis": USE_REDIS}

@app.post("/submit-job")
async def submit_job(
    bids_zip: UploadFile = File(...),
    participant_label: str = Form(...),
    modalities: str = Form(...),
    session_id: Optional[str] = Form(None),
    n_procs: int = Form(12),
    mem_gb: int = Form(48)
):
    job_id = str(uuid.uuid4())[:8]
    job_dir = UPLOAD_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    bids_path = job_dir / "bids_dataset.zip"

    with open(bids_path, "wb") as f:
        while chunk := await bids_zip.read(1024 * 1024):
            f.write(chunk)

    extract_dir = job_dir / "bids"
    with zipfile.ZipFile(bids_path, 'r') as zf:
        zf.extractall(extract_dir)

    result_dir = OUTPUT_ROOT / job_id
    set_status(job_id, {"status": "pending", "result": None})

    asyncio.create_task(
        run_mriqc_job(job_id, extract_dir, result_dir, participant_label, modalities, n_procs, mem_gb, session_id)
    )
    return {"job_id": job_id}

@app.get("/job-status/{job_id}")
def job_status(job_id: str):
    status = get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return status

@app.get("/download/{job_id}")
def download_result(job_id: str):
    result_zip = RESULT_ROOT / f"{job_id}.zip"
    if not result_zip.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(result_zip, filename=f"mriqc_results_{job_id}.zip", media_type="application/zip")

@app.delete("/delete-job/{job_id}")
def delete_job(job_id: str):
    try:
        shutil.rmtree(UPLOAD_ROOT / job_id, ignore_errors=True)
        shutil.rmtree(OUTPUT_ROOT / job_id, ignore_errors=True)
        (RESULT_ROOT / f"{job_id}.zip").unlink(missing_ok=True)
        clear_status(job_id)
        return {"status": "deleted"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Redis or memory status handlers
def set_status(job_id: str, status: dict):
    if USE_REDIS and rdb:
        rdb.set(f"mriqc:{job_id}", json.dumps(status))
    else:
        jobs[job_id] = status

def get_status(job_id: str):
    if USE_REDIS and rdb:
        raw = rdb.get(f"mriqc:{job_id}")
        return json.loads(raw) if raw else None
    return jobs.get(job_id)

def clear_status(job_id: str):
    if USE_REDIS and rdb:
        rdb.delete(f"mriqc:{job_id}")
    else:
        jobs.pop(job_id, None)

# Async MRIQC runner
async def run_mriqc_job(job_id, bids_dir, output_dir, participant_label, modalities, n_procs, mem_gb, session_id=None):
    try:
        cmd = [
            "docker", "run", "--rm",
            "--memory", f"{mem_gb}g", "--memory-swap", f"{mem_gb}g",
            "--cpus", str(n_procs),
            "-v", f"{bids_dir}:/data:ro",
            "-v", f"{output_dir}:/out",
            "nipreps/mriqc:22.0.6",
            "/data", "/out", "participant",
            "--participant_label", participant_label,
            "-m", *modalities.split(),
            "--nprocs", str(n_procs),
            "--omp-nthreads", "4",
            "--no-sub",
            "--verbose-reports"
        ]

        if session_id:
            cmd += ["--session-id", session_id]

        set_status(job_id, {"status": "running"})

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            set_status(job_id, {"status": "failed", "error": stderr.decode()})
            return

        zip_out = RESULT_ROOT / f"{job_id}.zip"
        shutil.make_archive(str(zip_out).replace(".zip", ""), 'zip', root_dir=output_dir)
        set_status(job_id, {"status": "complete", "result": str(zip_out)})

    except Exception as e:
        set_status(job_id, {"status": "failed", "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)
