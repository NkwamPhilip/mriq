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

app = FastAPI()

# Allow all origins (adjust this for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Global Paths
BASE_DIR = Path("/mnt")
UPLOAD_ROOT = BASE_DIR / "mriqc_upload"
OUTPUT_ROOT = BASE_DIR / "mriqc_output"
RESULT_ROOT = BASE_DIR / "mriqc_results"
os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(RESULT_ROOT, exist_ok=True)

# Track jobs
jobs: Dict[str, Dict] = {}

@app.get("/health")
async def health():
    return {"status": "ok", "message": "MRIQC backend is live"}

@app.post("/submit-job")
async def submit_job(
    bids_zip: UploadFile = File(...),
    participant_label: str = Form(...),
    modalities: str = Form(...),
    n_procs: int = Form(12),
    mem_gb: int = Form(48),
    session_id: Optional[str] = Form(None)
):
    job_id = str(uuid.uuid4())[:8]
    job_dir = UPLOAD_ROOT / job_id
    os.makedirs(job_dir, exist_ok=True)
    bids_path = job_dir / "bids_dataset.zip"

    try:
        with open(bids_path, "wb") as f:
            while chunk := await bids_zip.read(1024 * 1024):
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    extract_dir = job_dir / "bids"
    try:
        with zipfile.ZipFile(bids_path, 'r') as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file format")

    result_dir = OUTPUT_ROOT / job_id
    jobs[job_id] = {"status": "pending", "result": None}

    asyncio.create_task(run_mriqc_job(
        job_id=job_id,
        bids_dir=extract_dir,
        output_dir=result_dir,
        participant_label=participant_label,
        modalities=modalities,
        n_procs=n_procs,
        mem_gb=mem_gb,
        session_id=session_id
    ))

    return {"job_id": job_id, "status": "started"}

@app.get("/job-status/{job_id}")
async def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return jobs[job_id]

@app.get("/download/{job_id}")
async def download_result(job_id: str):
    result_zip = RESULT_ROOT / f"{job_id}.zip"
    if not result_zip.exists():
        raise HTTPException(status_code=404, detail="Result not ready or not found")
    return FileResponse(result_zip, filename=f"mriqc_results_{job_id}.zip")

@app.delete("/delete-job/{job_id}")
async def delete_job(job_id: str):
    # Clean all related directories
    for root in [UPLOAD_ROOT, OUTPUT_ROOT, RESULT_ROOT]:
        job_path = root / job_id
        if job_path.exists():
            shutil.rmtree(job_path, ignore_errors=True)

    zip_file = RESULT_ROOT / f"{job_id}.zip"
    if zip_file.exists():
        zip_file.unlink()

    if job_id in jobs:
        del jobs[job_id]

    return {"status": "deleted"}

# Async MRIQC Task
async def run_mriqc_job(
    job_id: str,
    bids_dir: Path,
    output_dir: Path,
    participant_label: str,
    modalities: str,
    n_procs: int,
    mem_gb: int,
    session_id: Optional[str]
):
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

        jobs[job_id]["status"] = "running"
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            jobs[job_id] = {
                "status": "failed",
                "error": stderr.decode()[-2000:]  # keep last 2KB
            }
            return

        # Package result
        zip_out = RESULT_ROOT / f"{job_id}.zip"
        shutil.make_archive(str(zip_out).replace(".zip", ""), 'zip', root_dir=output_dir)

        jobs[job_id] = {"status": "completed", "result": str(zip_out)}

    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
