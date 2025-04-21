# backend/main.py
import logging
import threading
import subprocess
import time
import sys
import os
import re
import pathlib
import shutil
from typing import Optional
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Request, Query, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TubeCoderAPI")

# --- Configuration ---
TEMP_BASE_DIR = pathlib.Path(os.getenv("TEMP_DATA_DIR", "./temp_youtube_data")).resolve()

# --- Import Cancellation Utils ---
try:
    from utils.cancellation_utils import (
        CancelledError,
        initialize_flag,
        set_cancel_flag,
        get_active_process,
        remove_active_process
    )
    CANCELLATION_ENABLED = True
except ImportError as e:
    logger.error(f"Failed to import cancellation utils: {e}. Cancellation endpoint will not function correctly.")
    CANCELLATION_ENABLED = False
    class CancelledError(Exception): pass
    def initialize_flag(sid): pass
    def set_cancel_flag(sid): pass
    def get_active_process(sid): return None
    def remove_active_process(sid): pass

# --- Import Agent Handler ---
try:
    from agent.tubecoder_agent import handle_agent_request, get_agent_executor
except ImportError as e:
    logger.error(f"Failed to import agent handler: {e}")
    def handle_agent_request(msg, sid):
        logger.error("Agent import failed. Returning error message.")
        return "ERROR: Agent failed to load.", sid or f"error_{uuid4()}"
    def get_agent_executor(): pass

# --- Lifespan Event for Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API starting up...")
    try:
        get_agent_executor()
        logger.info("Agent executor initialized successfully.")
    except Exception as e:
        logger.error(f"FATAL: Agent executor failed to initialize on startup: {e}")
    yield
    logger.info("API shutting down...")

# --- FastAPI App ---
app = FastAPI(title="TubeCoder API", lifespan=lifespan)

# --- CORS Middleware Configuration ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*", "POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# --- Request/Response Models ---
class ChatRequest(BaseModel):
    message: str
    sessionId: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: str

class CancelRequest(BaseModel):
    session_id: str = Field(..., description="The session ID of the request to cancel.")

# --- Chat Endpoint ---
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received chat request. Frontend sessionId: {request.sessionId}")
    session_id_to_use = request.sessionId

    # Generate new session ID if none provided
    is_new_session = False
    if not session_id_to_use:
        session_id_to_use = f"session_{uuid4()}"
        is_new_session = True
        logger.info(f"Generated new session ID for request: {session_id_to_use}")

    # Initialize flag for the session
    if CANCELLATION_ENABLED:
        initialize_flag(session_id_to_use)

    final_session_id = session_id_to_use

    try:
        # Use await for async handle_agent_request
        reply_text, returned_session_id = await handle_agent_request(request.message, session_id_to_use)
        final_session_id = returned_session_id

        return ChatResponse(reply=reply_text, session_id=final_session_id)

    except CancelledError as ce:
        logger.warning(f"Caught cancellation in API endpoint for session {session_id_to_use}: {ce}")
        cancelled_session_id = final_session_id
        if CANCELLATION_ENABLED and cancelled_session_id:
            initialize_flag(cancelled_session_id)
        return ChatResponse(reply="[Processing stopped by user]", session_id=cancelled_session_id)

    except Exception as e:
        logger.exception(f"Error in chat endpoint for session {session_id_to_use}: {e}")
        error_session_id = final_session_id
        if CANCELLATION_ENABLED and error_session_id:
            initialize_flag(error_session_id)
        raise HTTPException(status_code=500, detail=f"API Error: {e}", headers={"X-Error-Session-ID": error_session_id})

# --- Cancel Request Endpoint ---
@app.post("/api/cancel_request", status_code=status.HTTP_202_ACCEPTED)
async def cancel_request_endpoint(cancel_req: CancelRequest):
    print(f"--- ENTERED /api/cancel_request for session: {cancel_req.session_id} ---", flush=True)
    session_to_cancel = cancel_req.session_id
    print(f"--- Received cancel for session_id: {session_to_cancel} ---", flush=True)

    if not CANCELLATION_ENABLED:
        print("--- ERROR: Cancellation system not enabled ---", flush=True)
        raise HTTPException(status_code=500, detail="Cancellation system not initialized.")

    try:
        print(f"--- Calling set_cancel_flag for {session_to_cancel} ---", flush=True)
        set_cancel_flag(session_to_cancel)
        print(f"--- Returned from set_cancel_flag for {session_to_cancel} ---", flush=True)
    except Exception as e:
        print(f"--- ERROR calling set_cancel_flag: {e} ---", flush=True)
        raise HTTPException(status_code=500, detail="Failed to set cancel flag")

    print(f"--- Exiting /api/cancel_request for {session_to_cancel} ---", flush=True)
    return {"message": "Cancellation request processed.", "flag_set": True}

# --- Download Endpoint ---
@app.get("/api/download_project")
async def download_project_endpoint(
    session_id: str = Query(..., description="The unique session ID for the project to download."),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    logger.info(f"Received download request for session: {session_id}")

    if not session_id or not re.match(r"^[a-zA-Z0-9_\-]+$", session_id):
        logger.error(f"Invalid session_id format received: '{session_id}'")
        raise HTTPException(status_code=400, detail="Invalid session ID format.")

    project_dir = TEMP_BASE_DIR / session_id / "generated_project"
    logger.info(f"Looking for project directory at: {project_dir}")

    if not project_dir.is_dir():
        logger.error(f"Project directory not found for session {session_id} at {project_dir}")
        raise HTTPException(status_code=404, detail="Project not found. It might have expired or generation might have failed.")

    zip_filename_base = f"project_{session_id}"
    zip_output_path = project_dir.parent / f"{zip_filename_base}.zip"

    logger.info(f"Creating ZIP archive for '{project_dir}' at '{zip_output_path}'")

    try:
        archived_path_str = shutil.make_archive(
            base_name=str(zip_output_path.with_suffix('')),
            format='zip',
            root_dir=str(project_dir),
            base_dir='.'
        )
        zip_output_path = pathlib.Path(archived_path_str)

        if not zip_output_path.exists():
            raise FileNotFoundError("ZIP archive was not created by shutil.")

        logger.info(f"ZIP archive created successfully: {zip_output_path}")

        def cleanup_zip_file(temp_zip_path: pathlib.Path):
            try:
                logger.info(f"Running background task: Deleting temporary ZIP file {temp_zip_path}")
                temp_zip_path.unlink()
                logger.info(f"Successfully deleted temporary ZIP file: {temp_zip_path}")
            except Exception as e:
                logger.error(f"Error deleting temporary ZIP file {temp_zip_path} in background: {e}")

        background_tasks.add_task(cleanup_zip_file, zip_output_path)

        download_filename = f"tubecoder_project_{session_id}.zip"
        return FileResponse(
            path=str(zip_output_path),
            media_type='application/zip',
            filename=download_filename
        )

    except FileNotFoundError:
        logger.error(f"Project directory not found during zipping attempt: {project_dir}")
        raise HTTPException(status_code=404, detail="Project files not found. Generation may have failed.")
    except Exception as e:
        logger.exception(f"Error creating or sending ZIP archive for session {session_id}: {e}")
        if zip_output_path.exists():
            try:
                zip_output_path.unlink()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail="Failed to create project archive.")

# --- Root Endpoint ---
@app.get("/")
def read_root():
    return {"message": "TubeCoder API is running."}

# --- Run Server ---
if __name__ == "__main__":
    logger.info("Starting TubeCoder API server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)