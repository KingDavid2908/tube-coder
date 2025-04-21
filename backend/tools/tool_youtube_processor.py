#backend\tools\tool_youtube_processor.py
import os
import subprocess
import time
import math
import shutil
import re
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Audio processing
try:
    from pydub import AudioSegment
    from pydub.utils import make_chunks
except ImportError:
    logging.error("pydub not found. Install with 'pip install pydub'. Requires ffmpeg.")

# Gemini SDK
import google.generativeai as genai

# LangChain components
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    logging.error("FAISS not found. Install with 'pip install faiss-cpu langchain-community'.")
from langchain.docstore.document import Document

# --- Import cancellation helpers ---
try:
    from utils.cancellation_utils import check_cancellation, CancelledError
    CANCELLATION_ENABLED = True
except ImportError:
    tool_name = __name__.split('.')[-1]
    logger.error(f"Could not import cancellation checks from utils. Cancellation disabled for {tool_name}.")
    class CancelledError(Exception): pass
    def check_cancellation(session_id: str): pass
    CANCELLATION_ENABLED = False

try:
    from utils.llm_utils import call_gemini_with_retries
except ImportError:
    logger.warning("Could not use relative imports for utils, trying direct.")
    from llm_utils import call_gemini_with_retries
from typing import Union, Sequence, Any

# --- Configuration ---
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
MODEL_NAME = "gemini-2.5-flash-preview-04-17"
EMBEDDING_MODEL_NAME = "models/embedding-001"
MAX_CHUNK_DURATION_MS = int(os.getenv("MAX_AUDIO_CHUNK_HOURS", "5")) * 60 * 60 * 1000
YT_DLP_PATH = os.getenv("YT_DLP_PATH", "yt-dlp")

# --- Directory Setup ---
TEMP_BASE_DIR = Path(os.getenv("TEMP_DATA_DIR", "./temp_youtube_data")).resolve()
VECTORSTORE_BASE_PATH = Path(os.getenv("VECTORSTORE_DIR", "./rag_indices")).resolve()
Path("./temp_youtube_data").mkdir(parents=True, exist_ok=True)
Path("./rag_indices").mkdir(parents=True, exist_ok=True)
VECTORSTORE_PATH = VECTORSTORE_BASE_PATH / "youtube_faiss_index"

# --- Pydantic Input Schema ---
class YouTubeIndexArgs(BaseModel):
    youtube_url: str = Field(description="The URL of the YouTube video to process.")
    session_id: str = Field(description="A unique identifier for the current processing session.")

# --- Global Variables for Initialized Clients ---
_genai_configured = False
_transcription_model = None
_embeddings_model = None

def _initialize_gemini():
    global _genai_configured, _transcription_model, _embeddings_model
    if _genai_configured:
        return True
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not found in environment variables.")
        return False
    try:
        logger.info(f"Configuring Gemini ({MODEL_NAME}, {EMBEDDING_MODEL_NAME})...")
        genai.configure(api_key=GOOGLE_API_KEY)
        _transcription_model = genai.GenerativeModel(MODEL_NAME)
        _embeddings_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY
        )
        logger.info("Gemini Configured.")
        _genai_configured = True
        return True
    except Exception as e:
        logger.exception(f"Error configuring Gemini: {e}")
        _genai_configured = False
        return False

def get_video_metadata(url: str, session_id: str) -> dict | None:
    """Fetches video metadata as JSON using yt-dlp."""
    check_cancellation(session_id)
    command = [YT_DLP_PATH, '--dump-json', '--skip-download', '--no-warnings', '--quiet', url]
    logger.info(f"Fetching video metadata for: {url}")
    try:
        check_cancellation(session_id)
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='ignore',
            timeout=90
        )
        check_cancellation(session_id)
        metadata = json.loads(process.stdout)
        logger.info("Metadata fetched successfully.")
        return metadata
    except FileNotFoundError:
        logger.error(f"'{YT_DLP_PATH}' not found.")
        return None
    except subprocess.TimeoutExpired:
        logger.error("yt-dlp timed out fetching metadata.")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"yt-dlp failed metadata (code {e.returncode})\nstderr:\n{e.stderr}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON metadata: {e}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred fetching metadata: {e}")
        return None

def extract_info_from_description(description_text: str | None) -> dict:
    """Extracts links from the description text."""
    extracted_data = {"description_text": description_text or "", "links": []}
    if not description_text:
        logger.info("No description text provided.")
        return extracted_data
    url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
    try:
        links = url_pattern.findall(description_text)
        cleaned_links = sorted(list(set(link for link in links if '.' in link and len(link) > 5)))
        extracted_data["links"] = cleaned_links
        logger.info(f"Extracted {len(cleaned_links)} links from description.")
    except Exception as e:
        logger.warning(f"Error extracting links: {e}")
    return extracted_data

def get_audio_duration_ms(file_path: Path, session_id: str) -> int | None:
    """Gets audio duration in milliseconds using pydub."""
    check_cancellation(session_id)
    logger.info(f"Getting duration for {file_path.name}")
    try:
        audio = AudioSegment.from_file(file_path)
        check_cancellation(session_id)
        duration_ms = len(audio)
        logger.info(f"Audio duration: {duration_ms} ms")
        return duration_ms
    except Exception as e:
        logger.error(f"Error reading audio duration for {file_path.name}: {e}")
        logger.error("Ensure ffmpeg/ffprobe is installed and accessible by pydub.")
        return None

def split_audio(file_path: Path, chunk_length_ms: int, chunk_dir: Path, session_id: str) -> list[Path]:
    """Splits audio into chunks using pydub."""
    check_cancellation(session_id)
    logger.info(f"Splitting audio file: {file_path.name}")
    try:
        chunk_dir.mkdir(parents=True, exist_ok=True)
        audio = AudioSegment.from_file(file_path)
        check_cancellation(session_id)
        chunks = make_chunks(audio, chunk_length_ms)
        chunk_paths = []
        logger.info(f"Audio split into {len(chunks)} chunks. Exporting...")
        for i, hue in enumerate(chunks):
            check_cancellation(session_id)
            timestamp_suffix = int(time.time() * 1000)
            chunk_name = chunk_dir / f"chunk_{i+1}_{file_path.stem}_{timestamp_suffix}{file_path.suffix}"
            logger.info(f"Exporting {chunk_name.name}...")
            hue.export(chunk_name, format=file_path.suffix[1:])
            chunk_paths.append(chunk_name)
        logger.info("Chunk export complete.")
        return chunk_paths
    except Exception as e:
        logger.error(f"Error splitting audio {file_path.name}: {e}")
        logger.error("Ensure ffmpeg/ffprobe is installed and accessible by pydub.")
        return []

def transcribe_audio_chunk(audio_path: Path, model: genai.GenerativeModel, session_id: str) -> str | None:
    """Transcribes a single audio chunk using Gemini API."""
    check_cancellation(session_id)
    logger.info(f"Transcribing {audio_path.name}...")
    mime_type = f"audio/{audio_path.suffix[1:]}"
    audio_file_part = None
    try:
        check_cancellation(session_id)
        logger.info(f"Uploading {audio_path.name} to Gemini... (Session: {session_id})")
        audio_file_part = genai.upload_file(path=audio_path, mime_type=mime_type)
        logger.info(f"Upload status: {audio_file_part.name}, State: {audio_file_part.state.name}")

        while audio_file_part.state.name == "PROCESSING":
            check_cancellation(session_id)
            logger.info("Waiting for audio processing...")
            time.sleep(10)
            audio_file_part = genai.get_file(audio_file_part.name)
            logger.info(f"Current state: {audio_file_part.state.name}")

        check_cancellation(session_id)
        if audio_file_part.state.name == "FAILED":
            logger.error(f"Audio processing failed for {audio_path.name}.")
            if audio_file_part:
                try:
                    genai.delete_file(audio_file_part.name)
                    logger.info(f"Cleaned up failed file: {audio_file_part.name}")
                except Exception as del_e:
                    logger.warning(f"Could not delete failed file {audio_file_part.name}: {del_e}")
            return None
        if audio_file_part.state.name != "ACTIVE":
            logger.error(f"Audio file {audio_file_part.name} not active ({audio_file_part.state.name}). Cannot transcribe.")
            if audio_file_part:
                try:
                    genai.delete_file(audio_file_part.name)
                    logger.info(f"Cleaned up non-active file: {audio_file_part.name}")
                except Exception as del_e:
                    logger.warning(f"Could not delete non-active file {audio_file_part.name}: {del_e}")
            return None

        # --- Transcription Call with Retries ---
        transcript = None
        if audio_file_part:
            try:
                check_cancellation(session_id)
                logger.info(f"Attempting to transcribe {audio_path.name} using model {model.model_name}...")
                prompt_and_file = [
                    "Transcribe this audio accurately, including punctuation.",
                    audio_file_part
                ]
                response = call_gemini_with_retries(
                    model_input=prompt_and_file,
                    model_instance=model,
                    model_name=model.model_name,
                    request_options={'timeout': 720}
                )

                check_cancellation(session_id)
                extracted_text = None
                if hasattr(response, 'text') and response.text:
                    extracted_text = response.text
                elif hasattr(response, 'parts') and response.parts:
                    extracted_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))

                if extracted_text is not None:
                    logger.info("Transcription successful.")
                    transcript = extracted_text
                else:
                    logger.warning("Transcription call succeeded but no text found in response.")
                    transcript = ""

            except Exception as e:
                logger.error(f"Transcription failed for {audio_path.name} after retries or due to non-retryable error: {e}")
                transcript = None
        else:
            logger.error("Cannot transcribe, upload/processing failed earlier.")

        return transcript

    except CancelledError:
        logger.warning(f"Transcription cancelled for session {session_id} for chunk {audio_path.name}")
        if audio_file_part:
            try:
                genai.delete_file(audio_file_part.name)
            except Exception as del_e:
                logger.warning(f"Could not delete uploaded file {audio_file_part.name}: {del_e}")
        raise
    except Exception as e:
        logger.error(f"Error during transcription process for {audio_path.name}: {e}")
        if audio_file_part and hasattr(audio_file_part, 'name'):
            try:
                logger.info(f"Attempting cleanup for {audio_file_part.name} after error...")
                genai.delete_file(audio_file_part.name)
                logger.info(f"Cleaned up potentially failed upload: {audio_file_part.name}")
            except Exception as del_e:
                logger.warning(f"Could not delete file {audio_file_part.name} after error: {del_e}")
        return None
    finally:
        if audio_file_part and 'transcript' not in locals():
            try:
                genai.delete_file(audio_file_part.name)
                logger.info(f"Cleaned up uploaded file (in finally): {audio_file_part.name}")
            except Exception as del_e:
                logger.warning(f"Could not delete uploaded file {audio_file_part.name} in finally: {del_e}")

def download_audio(url: str, output_filename: str, download_dir: Path, session_id: str) -> Path | None:
    download_dir.mkdir(parents=True, exist_ok=True)
    output_template = download_dir / f"{output_filename}.%(ext)s"
    command = [
        YT_DLP_PATH,
        '-f', 'bestaudio[ext=m4a]/bestaudio',
        '-o', str(output_template),
        '--no-warnings',
        '--quiet',
        '--progress',
        '--verbose',
        url
    ]
    logger.info(f"Attempting download for {url} into {download_dir}")
    logger.debug(f"Executing command: {' '.join(command)}")
    process = None
    try:
        check_cancellation(session_id)
        logger.info(f"Starting download for {url} (Session: {session_id})")
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='ignore',
            timeout=1800
        )
        check_cancellation(session_id)
        logger.debug(f"yt-dlp stdout:\n{process.stdout}")
        destination_match = re.search(r"\[download\] Destination: (.*)", process.stdout)
        actual_file_path = None
        if destination_match:
            actual_file_path = Path(destination_match.group(1).strip())
            logger.info(f"Detected downloaded file via stdout: {actual_file_path}")
        if not actual_file_path or not actual_file_path.exists():
            logger.info("Scanning download directory for output file...")
            possible_files = list(download_dir.glob(f"{output_filename}.*"))
            valid_files = [p for p in possible_files if not p.name.endswith('.part')]
            if not valid_files:
                logger.error(f"yt-dlp finished successfully but no valid output file found matching '{output_filename}.*' in {download_dir}.")
                logger.error(f"yt-dlp stderr:\n{process.stderr}")
                return None
            actual_file_path = max(valid_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found downloaded file via directory scan: {actual_file_path}")
        if actual_file_path.exists() and actual_file_path.stat().st_size > 100:
            logger.info(f"Audio downloaded successfully: {actual_file_path.name}")
            return actual_file_path
        else:
            logger.error(f"Downloaded file {actual_file_path} is missing or empty.")
            return None
    except CancelledError:
        logger.warning(f"Download cancelled for session {session_id} before/after yt-dlp run.")
        raise
    except FileNotFoundError:
        logger.error(f"'{YT_DLP_PATH}' command not found. Ensure yt-dlp is installed and in PATH.")
        return None
    except subprocess.TimeoutExpired:
        logger.error("yt-dlp download timed out.")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"yt-dlp failed (code {e.returncode}).")
        logger.error(f"yt-dlp stderr:\n{e.stderr}")
        logger.error(f"yt-dlp stdout:\n{e.stdout}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected download error occurred: {e}")
        return None

def cleanup_session_files(session_dir: Path):
    if not session_dir.is_dir():
        logger.warning(f"Cleanup requested for non-existent session directory: {session_dir}")
        return
    logger.info(f"Cleaning up temporary session directory: {session_dir}")
    try:
        shutil.rmtree(session_dir)
        logger.info(f"Successfully removed {session_dir}")
    except Exception as e:
        logger.error(f"Error removing session directory {session_dir}: {e}")

def run_youtube_indexing(youtube_url: str, session_id: str) -> dict:
    logger.info(f"Starting YouTube indexing for URL: {youtube_url} (Session: {session_id})")
    session_dir = TEMP_BASE_DIR / session_id
    session_download_dir = session_dir / "downloads"
    session_chunk_dir = session_dir / "audio_chunks"
    combined_file_path = session_dir / "combined_analysis.md"
    
    try:
        check_cancellation(session_id)
        if not _initialize_gemini():
            return {"status": "error", "message": "Failed to initialize Gemini.", "combined_file_path": None}
        
        check_cancellation(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured session directory exists: {session_dir}")
        
        metadata = None
        description_section = "## Video Description\n\n[Could not retrieve video metadata]\n\n---\n"
        title = "Unknown Video"
        files_to_cleanup = []
        
        check_cancellation(session_id)
        logger.info("Fetching video metadata...")
        metadata = get_video_metadata(youtube_url, session_id)
        if metadata:
            title = metadata.get('title', 'Unknown Video')
            logger.info(f"Video Title: {title}")
            raw_description = metadata.get('description')
            if raw_description:
                description_info = extract_info_from_description(raw_description)
                description_section = f"## Video Description ({title})\n\n"
                description_section += description_info.get('description_text', '[No Text Extracted]')
                if description_info.get('links'):
                    description_section += "\n\n### Links Found in Description:\n"
                    description_section += "\n".join(f"- {link}" for link in description_info['links'])
                description_section += "\n\n---\n"
            else:
                description_section = f"## Video Description ({title})\n\n[No description provided]\n\n---\n"
        else:
            logger.warning("Could not retrieve video metadata.")

        check_cancellation(session_id)
        logger.info("Downloading audio...")
        safe_title = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in title)
        audio_filename_base = f"audio_{safe_title[:50]}" if title != "Unknown Video" else "downloaded_audio"
        downloaded_audio_path = download_audio(youtube_url, audio_filename_base, session_download_dir, session_id)
        if not downloaded_audio_path:
            check_cancellation(session_id)
            raise Exception("Audio download failed (not cancelled).")

        files_to_cleanup.append(downloaded_audio_path)
        check_cancellation(session_id)

        logger.info("Starting transcription process...")
        total_duration_ms = get_audio_duration_ms(downloaded_audio_path, session_id)
        if total_duration_ms is None:
            raise Exception("Could not get audio duration.")
        audio_files_to_process: list[Path] = []
        chunk_paths_to_delete: list[Path] = []
        if total_duration_ms > MAX_CHUNK_DURATION_MS:
            logger.info("Audio duration exceeds limit, chunking...")
            chunk_paths = split_audio(downloaded_audio_path, MAX_CHUNK_DURATION_MS, session_chunk_dir, session_id)
            if not chunk_paths:
                raise Exception("Audio chunking failed.")
            audio_files_to_process = chunk_paths
            chunk_paths_to_delete = chunk_paths
            files_to_cleanup.extend(chunk_paths_to_delete)
        else:
            logger.info("Audio duration within limit, processing directly.")
            audio_files_to_process = [downloaded_audio_path]

        full_transcript_parts = []
        transcription_successful = True
        if not _transcription_model:
            raise Exception("Transcription model not initialized.")
        logger.info("Starting transcription loop...")
        for i, audio_path in enumerate(audio_files_to_process):
            check_cancellation(session_id)
            logger.info(f"Transcribing audio file {i+1}/{len(audio_files_to_process)}: {audio_path.name}")
            transcript_part = transcribe_audio_chunk(audio_path, _transcription_model, session_id)
            if transcript_part is not None:
                full_transcript_parts.append(transcript_part)
            else:
                logger.warning(f"Transcription failed for chunk {audio_path.name}. Continuing if possible...")
            check_cancellation(session_id)

        if not full_transcript_parts:
            full_transcript = "[Audio transcription failed or yielded no text for all parts]"
            logger.warning(full_transcript)
        else:
            full_transcript = "\n\n".join(filter(None, full_transcript_parts))
            logger.info("Transcription complete.")

        check_cancellation(session_id)
        logger.info(f"Saving combined metadata and transcript to: {combined_file_path}")
        transcript_section = f"## Full Audio Transcript\n\n{full_transcript}"
        content_to_save = f"{description_section}\n{transcript_section}"
        try:
            check_cancellation(session_id)
            with open(combined_file_path, 'w', encoding="utf-8") as f:
                f.write(content_to_save)
            logger.info("Combined analysis file saved successfully.")
        except Exception as e:
            logger.error(f"Error saving combined analysis file to {combined_file_path}: {e}")
            raise Exception(f"Failed to save analysis file: {e}")

        # --- Session Specific Vector Store Path ---
        session_vectorstore_path = VECTORSTORE_BASE_PATH / session_id / "faiss_index"
        try:
            check_cancellation(session_id)
            session_vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured session vector store directory exists: {session_vectorstore_path.parent}")
        except Exception as e:
            logger.error(f"Failed to create session vectorstore directory {session_vectorstore_path.parent}: {e}")

        # --- Indexing for RAG (Session Specific) ---
        check_cancellation(session_id)
        logger.info(f"Attempting to index content for session '{session_id}' into index '{session_vectorstore_path}'...")
        if not _embeddings_model:
            raise Exception("Embeddings model not initialized before indexing.")
        if not content_to_save.strip():
            logger.warning(f"Content to index for session {session_id} is empty. Skipping RAG indexing.")
        elif not session_vectorstore_path.parent.exists():
            logger.error(f"Cannot save index because directory {session_vectorstore_path.parent} does not exist. Skipping RAG indexing.")
        else:
            try:
                logger.debug(f"Creating LangChain Document for session {session_id}.")
                doc = Document(page_content=content_to_save, metadata={"source_url": youtube_url, "processed_file": str(combined_file_path.resolve()), "session_id": session_id})

                logger.debug(f"Splitting document for session {session_id}.")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
                splits = text_splitter.split_documents([doc])

                if not splits:
                    logger.warning(f"Text splitting resulted in no chunks for session {session_id}. Cannot create session index.")
                else:
                    logger.info(f"Content split into {len(splits)} chunks for session {session_id} index.")

                    check_cancellation(session_id)
                    logger.info(f"Creating/Overwriting vector store for session {session_id} at {session_vectorstore_path}...")
                    vectorstore = FAISS.from_documents(splits, _embeddings_model)
                    check_cancellation(session_id)
                    logger.info(f"Saving session vector store to: {session_vectorstore_path}")
                    vectorstore.save_local(str(session_vectorstore_path))
                    logger.info(f"Session vector store saved successfully for session {session_id}.")
            except Exception as index_e:
                logger.exception(f"Error during session RAG indexing or vector store saving for session {session_id}: {index_e}")

        check_cancellation(session_id)
        logger.info(f"Cleaning up {len(files_to_cleanup)} temporary audio file(s)...")
        cleaned_audio_count = 0
        for f_path in files_to_cleanup:
            try:
                if f_path and f_path.exists():
                    f_path.unlink()
                    logger.debug(f"Deleted temp audio: {f_path.name}")
                    cleaned_audio_count += 1
            except Exception as e:
                logger.warning(f"Could not delete temp audio file {f_path}: {e}")
        logger.info(f"Cleaned up {cleaned_audio_count} audio files.")
        try:
            if session_download_dir.exists() and not any(session_download_dir.iterdir()):
                session_download_dir.rmdir()
            if session_chunk_dir.exists() and not any(session_chunk_dir.iterdir()):
                session_chunk_dir.rmdir()
        except OSError as e:
            logger.warning(f"Could not remove empty audio subdirs in {session_dir}: {e}")
        logger.info(f"YouTube indexing process completed successfully for session {session_id}.")
        return {
            "status": "success",
            "message": "Metadata, transcript saved. RAG index updated.",
            "combined_file_path": str(combined_file_path.resolve())
        }

    except CancelledError as ce:
        logger.warning(f"YouTube Indexing cancelled for session {session_id}: {ce}")
        try:
            cleanup_session_files(session_dir)
            logger.info(f"Cleaned up session directory: {session_dir}")
        except Exception as clean_e:
            logger.error(f"Failed to clean up session directory {session_dir}: {clean_e}")
        return {"status": "cancelled", "message": "Processing cancelled by user.", "combined_file_path": None}
    except Exception as e:
        logger.exception(f"Error during YouTube indexing main block for session {session_id}: {e}")
        return {
            "status": "error",
            "message": f"An error occurred during indexing: {e}",
            "combined_file_path": None
        }