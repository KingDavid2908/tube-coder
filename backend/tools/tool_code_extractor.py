# backend/tools/tool_code_extractor.py
import os
import subprocess
import json
import math
import re
import shutil
import time
import logging
import select
import datetime
import base64
import traceback
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import timedelta
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
import threading

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except ImportError:
    logging.error("LangChain Google GenAI components not found. Install with 'pip install langchain-google-genai langchain-core'")

try:
    from utils.llm_utils import call_gemini_with_retries
except ImportError:
    logger.warning("Could not use relative imports for utils, trying direct.")
    from llm_utils import call_gemini_with_retries

# --- Import Cancellation Helpers ---
try:
    from utils.cancellation_utils import check_cancellation, CancelledError
    CANCELLATION_ENABLED = True
except ImportError:
    tool_name = __name__.split('.')[-1]
    logger.error(f"Could not import cancellation checks from utils. Cancellation disabled for {tool_name}.")
    class CancelledError(Exception): pass
    def check_cancellation(session_id: str): pass
    CANCELLATION_ENABLED = False

# --- Configuration ---
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MULTIMODAL_MODEL_NAME = os.getenv("MULTIMODAL_MODEL", "gemini-2.5-flash-preview-04-17")

ANALYSIS_DURATION_MINUTES = int(os.getenv("ANALYSIS_DURATION_MINUTES", "20"))
ANALYSIS_STEP_MINUTES = int(os.getenv("ANALYSIS_STEP_MINUTES", "15"))
DOWNLOAD_DURATION_MINUTES = int(os.getenv("DOWNLOAD_DURATION_MINUTES", "5"))

analysis_duration_seconds = ANALYSIS_DURATION_MINUTES * 60
analysis_step_seconds = ANALYSIS_STEP_MINUTES * 60
download_duration_seconds = DOWNLOAD_DURATION_MINUTES * 60

TEMP_BASE_DIR = Path(os.getenv("TEMP_DATA_DIR", "./temp_youtube_data")).resolve()
VIDEO_SEGMENTS_SUBDIR = "video_segments"
YT_DLP_PATH = os.getenv("YT_DLP_PATH", "yt-dlp")
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

# --- Pydantic Input Schema ---
class CodeExtractionArgs(BaseModel):
    youtube_url: str = Field(description="The URL of the YouTube video to process.")
    combined_analysis_file_path: str = Field(description="The absolute path to the .md file where metadata/transcript exist and where extracted code analysis should be appended.")
    session_id: str = Field(description="A unique identifier for the current processing session.")
    time_ranges: Optional[List[str]] = Field(None, description="Optional list of specific time ranges (e.g., ['1:30-2:45', '15:00-16:10']) to analyze instead of the full video.")

    @field_validator('time_ranges')
    @classmethod
    def validate_time_ranges_v2(cls, v: Optional[List[str]]):
        if v is None:
            return v
        if not isinstance(v, list):
            logger.error('Validation Error: time_ranges must be a list of strings or null')
            raise ValueError('time_ranges must be a list of strings or null')
        pattern = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?\s*-\s*\d{1,2}:\d{2}(:\d{2})?$")
        for item in v:
            if not isinstance(item, str) or not pattern.match(item.strip()):
                logger.error(f"Validation Error: Invalid time range format: '{item}'. Expected like 'M:SS-M:SS'.")
                raise ValueError(f"Invalid time range format: '{item}'. Expected format like 'M:SS-M:SS'.")
        return v

# --- Global Variables for Initialized Clients ---
_multimodal_llm_initialized = False
_multimodal_llm = None

def _initialize_multimodal_llm():
    global _multimodal_llm_initialized, _multimodal_llm
    if _multimodal_llm_initialized:
        return True
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not found for multimodal LLM.")
        return False
    try:
        logger.info(f"Configuring Multimodal LLM ({MULTIMODAL_MODEL_NAME})...")
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        _multimodal_llm = ChatGoogleGenerativeAI(
            model=MULTIMODAL_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
            safety_settings=safety_settings,
            request_timeout=1200
        )
        logger.info("Multimodal LLM Initialized.")
        _multimodal_llm_initialized = True
        return True
    except Exception as e:
        logger.exception(f"Error configuring Multimodal LLM: {e}")
        _multimodal_llm_initialized = False
        return False

# --- Helper Functions ---
def normalize_youtube_url(url: str, session_id: str) -> Optional[str]:
    check_cancellation(session_id)
    url = url.strip()
    patterns = [
        r'(?:https?://)?(?:www.)?youtube.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www.)?youtube.com/embed/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?youtu.be/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www.)?youtube.com/shorts/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www.)?youtube.com/live/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www.)?youtube.com/v/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www.)?youtube.com/.?&v=([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www.)?youtube.com/.?#v=([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            standard_url = f"youtu.be/{video_id}"
            logger.info(f"Normalized URL '{url}' to '{standard_url}'")
            check_cancellation(session_id)
            return standard_url
    logger.warning(f"Could not normalize URL: '{url}'. Using provided URL.")
    if re.match(r'^https?://', url):
        check_cancellation(session_id)
        return url
    logger.error(f"Invalid URL format: {url}")
    return None

def get_video_duration(youtube_url: str, session_id: str) -> Optional[float]:
    check_cancellation(session_id)
    logger.info(f"Fetching duration for: {youtube_url}")
    command_duration = [YT_DLP_PATH, '--print', '%(duration)s', '--skip-download', '--quiet', '--no-warnings', youtube_url]
    command_json = [YT_DLP_PATH, '-J', '--skip-download', '--quiet', '--no-warnings', youtube_url]
    try:
        check_cancellation(session_id)
        result = subprocess.run(command_duration, capture_output=True, text=True, check=True, encoding='utf-8')
        check_cancellation(session_id)
        duration_str = result.stdout.strip()
        if duration_str and duration_str != "NA" and duration_str.replace('.', '', 1).isdigit():
            return float(duration_str)
        logger.warning(f"Direct duration failed ('{duration_str}'), trying JSON metadata...")
        check_cancellation(session_id)
        result = subprocess.run(command_json, capture_output=True, text=True, check=True, encoding='utf-8')
        check_cancellation(session_id)
        metadata = json.loads(result.stdout)
        if 'duration' in metadata and metadata['duration'] is not None:
            logger.info(f"Duration from JSON: {metadata['duration']}")
            return float(metadata['duration'])
        if metadata.get('is_live') or metadata.get('live_status') == 'is_live':
            logger.warning("Video is live or upcoming, duration unavailable.")
            return None
        logger.error("Could not extract duration from JSON metadata.")
        return None
    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.strip() if e.stderr else "[No Stderr]"
        logger.error(f"Error getting duration: {e}\nyt-dlp stderr: {stderr_output}")
        return None
    except FileNotFoundError:
        logger.error(f"'{YT_DLP_PATH}' not found. Ensure yt-dlp is installed.")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error getting duration: {e}")
        return None

def format_time(seconds: Optional[float]) -> str:
    if seconds is None or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        return "??:??:??"
    td = timedelta(seconds=int(round(seconds)))
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def cleanup_segments(session_temp_dir: Path, session_id: str) -> None:
    check_cancellation(session_id)
    video_dir = session_temp_dir / VIDEO_SEGMENTS_SUBDIR
    if video_dir.exists():
        logger.info(f"Removing temporary directory: {video_dir}")
        try:
            check_cancellation(session_id)
            shutil.rmtree(video_dir)
            check_cancellation(session_id)
            logger.info(f" Removed: {video_dir}")
        except PermissionError as e:
            logger.error(f"PermissionError removing {video_dir}: {e}")
            logger.error("Manually delete the directory after script completion.")
        except Exception as e:
            logger.exception(f"Error removing {video_dir}: {e}")

def parse_relative_timestamp(ts_string: str) -> Optional[float]:
    if not ts_string:
        return None
    parts = ts_string.strip().split(':')
    seconds = 0
    try:
        if len(parts) == 3:
            h, m = map(int, parts[:2])
            s = float(parts[2])
            seconds = h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m = int(parts[0])
            s = float(parts[1])
            seconds = m * 60 + s
        elif len(parts) == 1:
            seconds = float(parts[0])
        else:
            return None
    except ValueError:
        return None
    return seconds

def adjust_timestamps_in_response(response_text: str, segment_start_seconds: float, session_id: str) -> str:
    check_cancellation(session_id)
    if not response_text:
        return ""
    timestamp_pattern = re.compile(
        r"""
        (\b(?:Timestamp|at|around|approx.?|approx|Range|File:.*?, Timestamp)\s*[:~-]?\s*)?
        (
        (?: \d{1,2} : )?
        \d{1,2} :
        \d{1,2}
        (?: . \d{1,3} )?
        )
        (
        \s* [-–—] \s*
        (
        (?: \d{1,2} : )? \d{1,2} : \d{1,2} (?: . \d{1,3} )?
        )
        )?
        """, re.VERBOSE | re.IGNORECASE
    )
    adjusted_text = response_text
    offset = 0
    matches_found = 0
    try:
        for match in timestamp_pattern.finditer(response_text):
            check_cancellation(session_id)
            start_char_orig, end_char_orig = match.span()
            preceding_text = response_text[:start_char_orig]
            if preceding_text.count('```') % 2 != 0:
                continue
            start_char_adjusted = start_char_orig + offset
            end_char_adjusted = end_char_orig + offset
            original_match_text = adjusted_text[start_char_adjusted:end_char_adjusted]
            prefix = match.group(1) or ""
            start_rel_str = match.group(2)
            is_range = match.group(3) is not None
            end_rel_str = match.group(4) if is_range else None
            start_rel_sec = parse_relative_timestamp(start_rel_str)
            if start_rel_sec is not None:
                start_abs_sec = segment_start_seconds + start_rel_sec
                start_abs_str = format_time(start_abs_sec)
                if prefix.lower().startswith("file:") and "," in prefix:
                    file_part = prefix.split(",")[0]
                    replacement_text = f"{file_part}, Timestamp: {start_abs_str}"
                else:
                    clean_prefix = prefix if not prefix.lower().startswith("timestamp") else "Timestamp: "
                    replacement_text = f"{clean_prefix}{start_abs_str}"
                if is_range and end_rel_str:
                    end_rel_sec = parse_relative_timestamp(end_rel_str)
                    if end_rel_sec is not None:
                        if end_rel_sec < start_rel_sec:
                            logger.warning(f"End time {end_rel_str} before start {start_rel_str}. Using start only.")
                            replacement_text += f" - [End time invalid]"
                        else:
                            end_abs_sec = segment_start_seconds + end_rel_sec
                            end_abs_str = format_time(end_abs_sec)
                            replacement_text += f" - {end_abs_str}"
                    else:
                        replacement_text += f" - [Parse Error: {end_rel_str}]"
                        logger.warning(f"Could not parse end timestamp: {end_rel_str}")
                elif is_range:
                    replacement_text += " - [Missing End Time]"
                    logger.warning(f"Range missing end: {original_match_text}")
                adjusted_text = adjusted_text[:start_char_adjusted] + replacement_text + adjusted_text[end_char_adjusted:]
                offset += len(replacement_text) - len(original_match_text)
                matches_found += 1
            else:
                logger.warning(f"Could not parse start timestamp: {start_rel_str} in: {original_match_text}")
        check_cancellation(session_id)
    except Exception as e:
        logger.exception(f"Error adjusting timestamps: {e}")
        return f"[ERROR: Timestamp adjustment failed: {e}]\n---\nOriginal Response:\n{response_text}"
    logger.info(f"Adjusted {matches_found} timestamp(s) in response." if matches_found > 0 else "No timestamps adjusted.")
    return adjusted_text

def create_segment_prompt(segment_start_seconds: float, segment_end_seconds: float) -> str:
    segment_start_time_str = format_time(segment_start_seconds)
    segment_end_time_str = format_time(segment_end_seconds)
    prompt = f"""
    **Role:** You are an expert code analyzer and technical guide creator. Your primary function is to meticulously observe software development tutorial videos and document the **exact, unabridged steps** required to replicate the code and setup shown, with **maximum detail**. Maintain this high level of detail consistently for every segment analyzed; **DO NOT become less detailed or summarize more in later segments.**

    **Core Task:** Analyze the provided video segment ({segment_start_time_str} to {segment_end_time_str}) demonstrating a software development project. **First, internally perform a meticulous, step-by-step thought process identifying every single technical action and detail.** Then, based *only* on these observed actions within *this specific segment*, generate a clear, actionable, step-by-step guide for recreation, adhering **strictly and exhaustively** to the format exemplified below.

    **Input:** A video segment (the content spans {segment_start_time_str} to {segment_end_time_str} of the original video) featuring coding and related technical steps.

    **Context:**
    1.  **Primary:** The visual (code editor, terminal, browser dev tools) and audio information **within the current video segment ({segment_start_time_str} to {segment_end_time_str})**. Pay **pixel-perfect attention** to commands typed, code written/modified/explained, file interactions, configurations set, software used, and spoken instructions. **Assume nothing is trivial.**
    2.  **Secondary:** Use general development knowledge (languages, frameworks, package managers, git, CLI) to interpret actions accurately.
    3.  **Continuity:** Assume potential prior context exists, but generate output **only** for actions **explicitly performed or shown within this 15-minute segment**. The level of detail must remain maximal for *this specific segment*.

    **Output Requirements & Format:**

    **CRITICAL: Your output MUST start directly with the step-by-step guide content (like "Okay, let's recreate..."). The '## Segment Analysis...' header for the *current* segment ({segment_start_time_str} to {segment_end_time_str}) is provided immediately below the '---' separator before your expected output begins; DO NOT repeat or modify this header.** Adhere STRICTLY and EXHAUSTIVELY to the format and detail requirements demonstrated in the One-Shot Example below. Capture ALL technical details: commands, full code snippets, **full code snippets reflecting modifications (show the entire updated snippet/function, do not just describe the change)**, UI interactions, configurations, environment variables, library actions, and key explanations. DO NOT SUMMARIZE OR OMIT technical actions. DO NOT add conversational text outside the specified structure.** Provide ONLY the structured analysis based purely on the current segment. Output timestamps relative to the start of *this* segment. The steps can be as much as the example below or more steps than the example below but should be a better representation of the segment(i.e spread out along the segment to account for all seconds in the segment.)

    ---
    **One-Shot Example (Use this ONLY for formatting guidance below the header):**

    ## Segment Analysis ([Example Start Time 00:00:00] to [Example End Time 00:20:00])

    Okay, let's recreate the project shown in the video step-by-step. The goal is to build a LangChain application, deployed with LangServe, that takes an objective and uses OpenAI's prompt engineering guide as context to generate a new prompt template.

    **Assumptions:**

    *   You have Python 3.8+ and `pip` installed.
    *   You have access to the LangChain CLI (we'll install it).
    *   You have an OpenAI API key.
    *   You have (or can sign up for) a LangSmith account and API key for tracing (optional but recommended as shown in the video).

    ---

    **Step 1: Project Setup and Environment (0:00 - 2:08)**

    1.  **Create Project Directory (Implied before ~0:10):**
        *Purpose: Set up the main folder for the project.*
        *(Although not explicitly shown, the presenter starts in a specific directory structure. We assume you start by creating the base directory.)*
        ```bash
        mkdir openai-prompter-recreation
        cd openai-prompter-recreation
        ```

    2.  **Create and Activate Virtual Environment (Mentioned ~0:11):**
        *Purpose: Isolate project dependencies.*
        *(Presenter mentions using a fresh virtual environment, assumed created and activated beforehand.)*
        ```bash
        python -m venv venv
        # On macOS/Linux
        source venv/bin/activate
        # On Windows
        # .\\venv\\Scripts\\activate
        ```

    3.  **Install LangChain CLI and Base Dependencies (~0:25 - 0:37):**
        *Purpose: Install the necessary command-line interface and core LangServe libraries.*
        *(Presenter shows and runs this command to get necessary tools.)*
        ```bash
        pip install -U langchain-cli
        ```
        (This installs `langchain`, `langserve`, `fastapi`, `uvicorn`, etc., as dependencies)

    4.  **Bootstrap LangServe Project (~0:41 - 0:52):**
        *Purpose: Generate the standard LangServe application structure.*
        *(Presenter uses the CLI to create the app structure named 'openai-prompter' in the current directory, adding the openai package.)*
        ```bash
        langchain app new openai-prompter --package=openai
        ```
        *Action:* When asked `What package would you like to add?` the presenter presses Enter (skips adding more). (Timestamp: 0:48 relative to example segment)
        *Action:* Navigate into the created directory. (Timestamp: 0:53 relative to example segment)
        ```bash
        cd openai-prompter
        ```
        This command creates the basic structure (`app/`, `packages/`, `pyproject.toml`, etc.) within the `openai-prompter` directory.

    5.  **(Optional but Recommended) Configure LangSmith (~1:08 - 2:08):**
        *Purpose: Enable tracing and debugging via LangSmith by setting environment variables.*
        Set these environment variables in your terminal *before* running `langchain serve`. You get these from the LangSmith settings page (shown around ~1:39 - ~1:54 in the video). The presenter copies these from the LangSmith UI 'Setup' tab around ~1:55.
        ```bash
        export LANGCHAIN_TRACING_V2=true
        export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
        export LANGCHAIN_API_KEY="your_langsmith_key_example"
        export LANGCHAIN_PROJECT="openai-prompter-recreation"
        ```
        Also, ensure your OpenAI API key is set as an environment variable:
        ```bash
        export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx_example"
        ```
        *Action:* Presenter creates a new LangSmith project named 'openai-prompter' via the UI. (Timestamp: ~1:41 relative to example segment)

    ---

    **Step 2: Prepare Context Data (Prompt Engineering Guide) (2:35 - 2:45)**

    1.  **Create the Text File (~2:44):**
        *Purpose: Create the file to store the prompt engineering guide text.*
        *(The presenter is now in a Jupyter Notebook environment for development initially.)* Create a file named `openai-prompting.txt` locally.
        *(Action: Presenter uses Jupyter file browser or similar to create the file)*

    2.  **Add Content (~2:35 - ~2:45):**
        *Purpose: Populate the text file with the context needed for the prompt generation chain.*
        Open `openai-prompting.txt` in a text editor. The presenter selects and copies the *entire* text content from the OpenAI Prompt Engineering guide web page (`https://platform.openai.com/docs/guides/prompt-engineering`) and pastes it into the file.
        *Action:* Copy the full text from the specified URL and paste it into `openai-prompting.txt`.

        *Example Snippet (using the real guide content is required for the app to work as intended):*
        ```txt
        Prompt engineering
        This guide shares strategies and tactics for getting better results from large language models ...
        ... [Rest of the entire guide text] ...
        ```

    ---

    **Step 3: Develop Chain Logic (Jupyter Notebook) (3:33 - 13:07)**

    1.  **Load Context Text (~3:36 - 3:50):**
        *Purpose: Read the guide content into a Python variable.*
        ```python
        # Timestamp: ~3:36
        # File Path Guess: openai-prompt-engineering.ipynb (jupyter)
        # Purpose: Load the prompt engineering guide text from the file into a variable.
        with open("openai-prompting.txt", "r") as f:
            text = f.read()
        # Presenter prints the first 200 chars to verify (~3:55)
        print(text[:200])
        ```

    2.  **Import LangChain Components (~4:00 - 4:27):**
        *Purpose: Import necessary classes for building the chain.*
        ```python
        # Timestamp: ~4:00 - 4:27
        # File Path Guess: openai-prompt-engineering.ipynb (jupyter)
        # Purpose: Import necessary modules from LangChain and OpenAI.
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parser import StrOutputParser
        from langchain_openai import ChatOpenAI
        ```

    3.  **Define Initial Prompt Template (~5:04 - 5:27):**
        *Purpose: Create the first version of the prompt template string.*
        ```python
        # Timestamp: ~5:04
        # File Path Guess: openai-prompt-engineering.ipynb (jupyter)
        # Purpose: Define the main prompt template that instructs the LLM how to generate a new prompt template.
        template = "
        Based on the above instructions, help me write a good prompt TEMPLATE.

        This template should be a python f-string. It can take in any number of variables depending on my objective.
        Return your answer in the following format:

        ```prompt
        ...
        ```

        This is my objective:
        {{{{objective}}}}
        "
        ```

    4.  **Handle Potential Formatting Errors with Partial (~5:31 - 5:40, revisited ~8:30 - ~9:14):**
        *Purpose: Pre-fill the 'text' variable to avoid errors if the guide text contains curly braces.*
        *(Presenter initially encounters KeyError because the guide text has its own `{...}` which conflict with PromptTemplate's variables. They later fix this using `prompt.partial`)*
        ```python
        # Timestamp: ~8:30 - 9:14
        # File Path Guess: openai-prompt-engineering.ipynb (jupyter)
        # Purpose: Create the PromptTemplate object and pre-fill the 'text' variable.
        base_prompt = PromptTemplate.from_template(template)
        prompt = base_prompt.partial(text=text)
        ```

    5.  **Define Model and Parser (~6:00 - 6:10):**
        *Purpose: Instantiate the LLM and the output parser.*
        ```python
        # Timestamp: ~6:00
        # File Path Guess: openai-prompt-engineering.ipynb (jupyter)
        # Purpose: Define the specific OpenAI model and output parser.
        model = ChatOpenAI()
        parser = StrOutputParser()
        ```

    6.  **Create Initial Chain (~6:10):**
        *Purpose: Combine the components into a runnable sequence using LCEL.*
        ```python
        # Timestamp: ~6:10
        # File Path Guess: openai-prompt-engineering.ipynb (jupyter)
        # Purpose: Create the runnable LangChain sequence.
        chain = prompt | model | parser
        ```

    7.  **Test Chain and Encounter KeyError (~7:40 - 8:30):**
        *Purpose: Invoke the chain to see the output and identify issues.*
        *Action:* Presenter invokes the chain with an objective.
        ```python
        # Timestamp: ~8:07
        # File Path Guess: openai-prompt-engineering.ipynb (jupyter)
        task = "answer a question based on context provided, and ONLY on that context."
        # This invocation fails due to unescaped braces in guide text
        # chain.invoke({{"objective": task}})
        ```

    8.  **Refine Model Choice & Temperature (~9:50 - 10:11):**
        *Purpose: Switch to a model with longer context and ensure deterministic output.*
        *(Presenter realizes default model context might be too short.)*
        ```python
        # Timestamp: ~10:05
        # File Path Guess: openai-prompt-engineering.ipynb (jupyter)
        # Purpose: Update model to GPT-4 Turbo preview.
        model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
        ```

    9.  **Recreate Chain with Partial Prompt and New Model (~11:30 - 11:41):**
        *Purpose: Combine the corrected prompt and updated model.*
        ```python
        # Timestamp: ~11:30 - 11:41
        # File Path Guess: openai-prompt-engineering.ipynb (jupyter)
        # Purpose: Recreate the chain.
        chain = prompt | model | parser
        ```

    10. **Refine Prompt Template Instruction (~11:44, ~12:45):**
        *Purpose: Improve the instruction for better formatting.*
        *(Presenter modifies template based on playground output.)*
        ```python
        # Timestamp: ~11:44
        # File Path Guess: openai-prompt-engineering.ipynb (jupyter)
        # Purpose: Modify template for clearer instructions.
        template_string = "
        Based on the above instructions, help me write a good prompt TEMPLATE.

        Notably, this prompt TEMPLATE expects that additional information will be provided later.
        When you have enough information to create a good prompt, return the prompt.

        Instructions for a good prompt:
        ---------------------
        {{{{text}}}}
        ---------------------

        This is my objective:
        {{{{objective}}}}

        Return only the python f-string
        "
        base_prompt = PromptTemplate.from_template(template_string)
        prompt = base_prompt.partial(text=guide_text)
        chain = prompt | model | parser
        ```

    11. **Test Refined Chain with Streaming (~11:57 - 12:26):**
        *Purpose: Test the final chain configuration using streaming output.*
        ```python
        # Timestamp: ~11:57 - 12:26
        # File Path Guess: openai-prompt-engineering.ipynb (jupyter)
        # Purpose: Test the refined chain.
        task = "answer a question based on context provided, and ONLY on that context."
        for token in chain.stream({{"objective": task}}):
            print(token, end="")
        ```

    ---

    **Step 4: Configure the LangServe Server (in `app/server.py`) (13:08 - 13:18)**

    1.  **Edit `app/server.py` (~13:08):**
        Open the existing `app/server.py` file generated by `langchain app new`.

    2.  **Modify the Code (~13:08 - ~13:18):**
        *Purpose: Set up the FastAPI web server and expose the chain as an API endpoint.*
        Make sure it looks like this:
        ```python
        # Timestamp: ~13:08
        # File Path Guess: app/server.py (api)
        # Purpose: Set up the FastAPI server.
        from fastapi import FastAPI
        from fastapi.responses import RedirectResponse
        from langserve import add_routes
        from app.chain import chain as openai_prompter_chain

        app = FastAPI(
            title="LangChain Server",
            version="1.0",
            description="A simple api server using Langchain's Runnable interfaces",
        )

        @app.get("/")
        async def redirect_root_to_docs():
            return RedirectResponse("/docs")

        add_routes(
            app,
            openai_prompter_chain,
            path="/prompter",
        )

        if __name__ == "__main__":
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000)
        ```

    ---

    **Step 5: Run and Test Locally (13:20 - 14:40)**

    1.  **Run the Server (~13:20):**
        *Purpose: Start the local development server.*
        ```bash
        langchain serve
        ```

    2.  **Access the Playground (~13:28):**
        *Purpose: Open the web-based interface to interact with the chain.*
        Open browser to:
        `http://127.0.0.1:8000/prompter/playground/`

    3.  **Test (~13:34 - ~14:34):**
        *Purpose: Send input and verify output.*
        *   Enter task: `answer a question based on context provided, and ONLY on that context.` (Timestamp: ~13:36)
        *   Click "Start". (Timestamp: ~13:38)
        *   Observe streaming output. (Timestamp: ~13:40 - ~14:34)

    4.  **Check LangSmith (Optional) (~14:38 - 14:59):**
        *Purpose: Trace execution flow.*
        Go to LangSmith project (`openai-prompter-recreation`). Trace shown ~14:43.

    ---

    **Step 6: Install OpenAI Dependency (~16:57 - 17:06)**
        *Purpose: Add OpenAI library explicitly.*
        ```bash
        poetry add openai
        ```

    ---

    **Step 7: Start LangSmith Deployment Setup (~18:09 - 19:59)**
        *Purpose: Begin configuring deployment via LangSmith connected to GitHub.*
        *(Action)*: Navigate to LangSmith UI. (Timestamp: ~18:09)
        *(Action)*: Switch to 'LangChain Inc.' organization. (Timestamp: ~19:41)
        *(Action)*: Click 'Deployments' section. (Timestamp: ~19:44)
        *(Action)*: Click '+ New Deployment' button. (Timestamp: ~19:47)

    You have now recreated the core functionality, local testing, dependency addition, and the *start* of the deployment configuration shown in the video up to the 20:00 mark!

    ---
    **Final Reminder:** Your output MUST be maximally exhaustive and focused *only* on the video segment being processed. Think step-by-step before writing. Adhere **strictly** to the specified Markdown format shown in the example above, including **timestamps**, **file paths (guessed if necessary)**, **purpose comments**, and capturing **all code modifications fully and exactly**. Use placeholders like `your_api_key_example` only within the example description itself, not when describing actual user actions that require real keys. Maintain the same high level of detail for ALL segments. Each steps in each segment must be well detailed.

    If the *entire segment* contains absolutely no relevant technical information for recreation (e.g., only conceptual slides or conversational intro/outro), respond ONLY with the exact phrase:
    `No relevant technical information found in this segment.`
    **Highly Important:**  Each steps in each segment must be well detailed and based on just the video segment. Don't hallucinate. Write down every code shown in the video segment. Don't summarize or skip any code.

    ---
    ## Segment Analysis ({segment_start_time_str} to {segment_end_time_str})
    """
    return prompt

def download_segment(youtube_url: str, start_seconds: float, end_seconds: float, output_filename: str, session_video_dir: Path, session_id: str) -> Optional[str]:
    check_cancellation(session_id)
    start_time = format_time(start_seconds)
    end_time = format_time(end_seconds)
    session_video_dir.mkdir(parents=True, exist_ok=True)
    base_output_name = output_filename
    output_path_template = str(session_video_dir / f"{base_output_name}.%(ext)s")
    time_range = f"*{start_time}-{end_time}"
    format_selection = 'bestvideo[height<=?480]+bestaudio/best[height<=?480]'
    concurrent_fragments = 8

    command = [
        YT_DLP_PATH,
        '--no-warnings',
        '--download-sections', time_range,
        '-f', format_selection,
        '--concurrent-fragments', str(concurrent_fragments),
        '-o', output_path_template,
        '--quiet',
        '--verbose',
        youtube_url
    ]
    segment_label = base_output_name
    logger.info(f"Downloading segment: {segment_label} ({start_time} to {end_time}) for session {session_id}")
    logger.debug(f"Executing command: {' '.join(command)}")
    process = None
    actual_output_path = None

    try:
        logger.debug(f"CHECKPOINT CE_DS1: Before start for {session_id}")
        check_cancellation(session_id)
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='ignore',
            timeout=300
        )
        logger.debug(f"CHECKPOINT CE_DS2: After subprocess.run for {session_id}")
        check_cancellation(session_id)

        stdout_output = process.stdout
        stderr_output = process.stderr
        if stderr_output:
            logger.debug(f"yt-dlp stderr for {segment_label}:\n{stderr_output}")

        dest_pattern = re.compile(
            r"\[(?:download|Merger|ExtractAudio|VideoConvertor|MoveFiles)\]\s+"
            r"(?:Destination:\s*|Merging formats into|Extracting audio to|Converting video to|Moving destination file to)\s+"
            r"\"?(.+?)\"?$",
            re.MULTILINE
        )
        dest_match = dest_pattern.search(stdout_output) or dest_pattern.search(stderr_output)
        if dest_match:
            actual_output_path = dest_match.group(1).strip()
            logger.info(f"Detected output file via stdout/stderr: {Path(actual_output_path).name}")
        else:
            logger.warning("Could not detect output file path via output. Scanning directory...")
            check_cancellation(session_id)
            possible_files = [f for f in session_video_dir.glob(f"{base_output_name}*") if not f.suffix in ('.part', '.ytdl', '.temp')]
            if possible_files:
                actual_output_path = str(max(possible_files, key=lambda x: x.stat().st_mtime))
                logger.info(f"Found output file via directory scan: {Path(actual_output_path).name}")
            else:
                logger.warning("No output file found via scan.")

        check_cancellation(session_id)
        if actual_output_path and Path(actual_output_path).exists() and Path(actual_output_path).stat().st_size > 100:
            logger.info(f"Verified output file: {Path(actual_output_path).name}")
            return actual_output_path
        else:
            logger.error(f"Verification failed for downloaded segment: {actual_output_path}")
            return None

    except CancelledError:
        logger.warning(f"Download segment cancelled for session {session_id}.")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"Download segment failed for {segment_label}: {e}\nstderr:\n{e.stderr}\nstdout:\n{e.stdout}")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout downloading segment {segment_label}.")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error downloading segment {segment_label}: {e}")
        return None
    finally:
        check_cancellation(session_id)
        part_extensions = ['.part', '.mp4.part', '.mkv.part', '.webm.part']
        for ext in part_extensions:
            part_file = session_video_dir / f"{base_output_name}{ext}"
            if part_file.exists():
                logger.info(f"Cleaning up: {part_file.name}")
                try:
                    time.sleep(0.2)
                    part_file.unlink()
                except Exception as e:
                    logger.warning(f"Could not remove {part_file.name}: {e}")

def concatenate_segments(file_parts: List[str], output_path: str, session_temp_dir: Path, session_id: str) -> bool:
    check_cancellation(session_id)
    list_filename = str(session_temp_dir / VIDEO_SEGMENTS_SUBDIR / f"ffmpeg_list_{Path(output_path).stem}.txt")
    logger.info(f"Concatenating {len(file_parts)} parts to {Path(output_path).name}")
    try:
        check_cancellation(session_id)
        subprocess.run([FFMPEG_PATH, '-version'], capture_output=True, check=True)
        check_cancellation(session_id)
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.error(f"'{FFMPEG_PATH}' not found or failed. Cannot concatenate.")
        return False
    try:
        check_cancellation(session_id)
        with open(list_filename, "w", encoding='utf-8') as f:
            for part_path in file_parts:
                if not Path(part_path).exists() or Path(part_path).stat().st_size == 0:
                    logger.error(f"Missing or empty part: {part_path}")
                    return False
                relative_filename = Path(part_path).name.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{relative_filename}'\n")
        check_cancellation(session_id)
        command = [
            FFMPEG_PATH,
            '-f', 'concat',
            '-safe', '0',
            '-i', list_filename,
            '-c', 'copy',
            '-y',
            '-loglevel', 'warning',
            output_path
        ]
        logger.info(f"Running ffmpeg: {' '.join(command)}")
        check_cancellation(session_id)
        process = subprocess.run(command, capture_output=True, text=True, check=False, encoding='utf-8', errors='replace', timeout=600)
        check_cancellation(session_id)
        if process.returncode != 0:
            logger.error(f"ffmpeg failed with code {process.returncode}\nstderr:\n{process.stderr}\nstdout:\n{process.stdout}")
            return False
        check_cancellation(session_id)
        if not Path(output_path).exists() or Path(output_path).stat().st_size < 100:
            logger.error(f"ffmpeg succeeded but output invalid: {output_path}")
            return False
        logger.info(f" Concatenated to {Path(output_path).name}")
        return True
    except CancelledError:
        logger.warning(f"Concatenation cancelled for session {session_id}.")
        raise
    except subprocess.TimeoutExpired:
        logger.error(f"ffmpeg timeout for {output_path}")
        return False
    except Exception as e:
        logger.exception(f"Error concatenating: {e}")
        return False
    finally:
        check_cancellation(session_id)
        if Path(list_filename).exists():
            try:
                Path(list_filename).unlink()
            except Exception as e:
                logger.warning(f"Could not remove {list_filename}: {e}")

def analyze_combined_segment(video_path: str, analysis_start_s: float, analysis_end_s: float, llm_instance: Any, session_id: str) -> str:
    logger.debug(f"CHECKPOINT CE_ACS1: Start analyze_combined_segment for {session_id}")
    check_cancellation(session_id)
    logger.info(f"Analyzing: {Path(video_path).name} ({format_time(analysis_start_s)}-{format_time(analysis_end_s)})")
    analysis_result_text = f"[ERROR: Analysis Failed for Block {format_time(analysis_start_s)}-{format_time(analysis_end_s)}]"
    try:
        check_cancellation(session_id)
        video_bytes = b''
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()
        check_cancellation(session_id)
        if not video_bytes:
            raise ValueError("Read 0 bytes from video file.")
        _, ext = os.path.splitext(video_path)
        ext = ext.lower().lstrip('.')
        mime_type = f"video/{ext}" if ext in ['mp4', 'mkv', 'webm', 'mov', 'avi'] else "video/mp4"
        logger.debug(f"Using MIME type: {mime_type}")
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        check_cancellation(session_id)
        prompt_text = create_segment_prompt(analysis_start_s, analysis_end_s)
        invoke_input = None
        if isinstance(llm_instance, ChatGoogleGenerativeAI):
            logger.debug("Constructing HumanMessage for ChatGoogleGenerativeAI")
            message = HumanMessage(content=[
                {"type": "text", "text": prompt_text},
                {"type": "media", "mime_type": mime_type, "data": video_base64},
            ])
            invoke_input = [message]
        else:
            logger.debug("Using fallback input for non-ChatGoogleGenerativeAI")
            invoke_input = [{"type": "text", "text": prompt_text}]
        logger.debug(f"CHECKPOINT CE_ACS2: Before LLM call for {session_id}")
        check_cancellation(session_id)
        if invoke_input is None:
            raise ValueError("LLM input format undetermined.")

        analysis_result_text = None
        last_exception_detail = "Unknown analysis failure"

        if invoke_input:
            try:
                logger.info(f"Attempting LLM analysis for {format_time(analysis_start_s)}-{format_time(analysis_end_s)}...")
                response = call_gemini_with_retries(
                    model_input=invoke_input,
                    model_instance=llm_instance,
                    model_name=MULTIMODAL_MODEL_NAME
                )
                logger.debug(f"CHECKPOINT CE_ACS3: After LLM call for {session_id}")
                check_cancellation(session_id)
                llm_response_content = None
                if isinstance(response, AIMessage):
                    llm_response_content = response.content.strip() if isinstance(response.content, str) else None
                    if not llm_response_content:
                        logger.warning("LLM AIMessage content is empty.")
                        llm_response_content = "[INFO: Analysis resulted in empty content]"
                    metadata = getattr(response, 'response_metadata', {})
                    if metadata.get('prompt_feedback', {}).get('block_reason'):
                        block_reason = metadata['prompt_feedback']['block_reason']
                        logger.error(f"LLM Call Blocked (via LangChain metadata). Reason: {block_reason}")
                        raise Exception(f"API Call Blocked (LangChain): {block_reason}")
                elif isinstance(response, str):
                    llm_response_content = response.strip()
                    if not llm_response_content:
                        logger.warning("LLM response was empty string.")
                        llm_response_content = "[INFO: Analysis resulted in empty content]"
                else:
                    logger.error(f"Unexpected response type from invoke: {type(response)}")
                    raise ValueError(f"Unexpected response type: {type(response)}")

                check_cancellation(session_id)
                if llm_response_content:
                    if "No relevant technical information found" in llm_response_content:
                        analysis_result_text = llm_response_content
                        logger.info(f"No relevant info found by LLM.")
                    else:
                        logger.info(f"Adjusting timestamps...")
                        analysis_result_text = adjust_timestamps_in_response(llm_response_content, analysis_start_s, session_id)

            except CancelledError:
                logger.warning(f"Analysis cancelled within analyze_combined_segment for {session_id}")
                raise
            except Exception as e:
                logger.error(f"LLM analysis failed for {analysis_start_s}-{analysis_end_s} after retries or due to non-retryable error: {e}")
                analysis_result_text = None
                last_exception_detail = f"{e}"

        else:
            logger.error("Cannot invoke LLM, input construction failed earlier.")
            return f"## Segment Analysis ({format_time(analysis_start_s)} - {format_time(analysis_end_s)})\n\n[ERROR: Failed to construct LLM input]"

        check_cancellation(session_id)
        segment_header = f"## Segment Analysis ({format_time(analysis_start_s)} - {format_time(analysis_end_s)})"
        if analysis_result_text is not None:
            return f"{segment_header}\n{analysis_result_text}"
        else:
            return f"{segment_header}\n\n[ERROR: Analysis failed for this segment after retries]: {last_exception_detail}]"

    except CancelledError:
        logger.warning(f"Analysis cancelled for session {session_id}.")
        raise
    except Exception as e:
        logger.exception(f"Analysis error for {analysis_start_s}-{analysis_end_s}: {e}")
        analysis_result_text = f"[ERROR: Analysis Exception - {type(e).__name__}]"
        segment_header = f"## Segment Analysis ({format_time(analysis_start_s)} - {format_time(analysis_end_s)})"
        return f"{segment_header}\n{analysis_result_text}"

# --- Core Logic Function ---
def process_video_for_code(
    youtube_url: str,
    combined_analysis_file: Path,
    session_id: str,
    target_time_ranges: Optional[List[str]] = None
) -> bool:
    check_cancellation(session_id)
    session_temp_dir = TEMP_BASE_DIR / session_id
    session_video_dir = session_temp_dir / VIDEO_SEGMENTS_SUBDIR
    session_video_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processing video for code (Efficient Download). Session Temp: {session_temp_dir}")

    if not _multimodal_llm:
        logger.error("Multimodal LLM not initialized.")
        return False

    check_cancellation(session_id)
    total_duration = get_video_duration(youtube_url, session_id)
    check_cancellation(session_id)
    if total_duration is None:
        logger.error("Failed to get video duration. Aborting code extraction.")
        return False
    logger.info(f"Video Duration: {format_time(total_duration)} ({total_duration:.2f} seconds)")

    analysis_segments_times: List[Dict[str, float]] = []
    is_targeted_analysis = bool(target_time_ranges)

    if is_targeted_analysis:
        logger.info(f"Performing targeted analysis for time ranges: {target_time_ranges}")
        def parse_hms_to_seconds(time_str: str) -> float:
            time_str = time_str.strip()
            parts = list(map(float, time_str.split(':')))
            if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
            if len(parts) == 2: return parts[0] * 60 + parts[1]
            if len(parts) == 1: return parts[0]
            raise ValueError(f"Invalid time format: {time_str}")
        try:
            check_cancellation(session_id)
            for time_range_str in target_time_ranges:
                check_cancellation(session_id)
                start_str, end_str = map(str.strip, time_range_str.split('-'))
                start_s = parse_hms_to_seconds(start_str)
                end_s = parse_hms_to_seconds(end_str)
                start_s = max(0.0, start_s)
                end_s = min(total_duration, end_s)
                if end_s > start_s + 0.1:
                    analysis_segments_times.append({'start': start_s, 'end': end_s})
                else:
                    logger.warning(f"Skipping invalid or zero-duration time range: {time_range_str}")
            check_cancellation(session_id)
        except Exception as e:
            logger.error(f"Error parsing target_time_ranges '{target_time_ranges}': {e}. Aborting.")
            return False
        if not analysis_segments_times:
            logger.error("No valid analysis segments derived from target_time_ranges. Aborting.")
            return False
        logger.info(f"Targeted analysis blocks: {analysis_segments_times}")
    else:
        logger.info("Performing full video analysis with overlapping segments.")
        current_start_s = 0.0
        while True:
            check_cancellation(session_id)
            start_s = current_start_s
            if start_s >= total_duration: break
            end_s = min(start_s + analysis_duration_seconds, total_duration)
            if end_s > start_s + 0.5:
                analysis_segments_times.append({'start': start_s, 'end': end_s})
            else:
                if len(analysis_segments_times) > 0: break
            current_start_s += analysis_step_seconds
            if analysis_segments_times and analysis_segments_times[-1]['end'] >= total_duration: break
        check_cancellation(session_id)
        if not analysis_segments_times:
            logger.warning("No analysis segments generated for full video (video might be too short).")
            return True

    unique_download_chunks = set()
    logger.info("Calculating unique download chunks needed across all analysis blocks...")
    for analysis_block in analysis_segments_times:
        check_cancellation(session_id)
        analysis_start_s = analysis_block['start']
        analysis_end_s = analysis_block['end']
        first_part_start_s = math.floor(analysis_start_s / download_duration_seconds) * download_duration_seconds
        last_part_end_s_ceil = math.ceil(analysis_end_s / download_duration_seconds) * download_duration_seconds
        current_part_start = first_part_start_s
        while current_part_start < analysis_end_s and current_part_start < total_duration:
            check_cancellation(session_id)
            part_start_s = current_part_start
            part_end_s = min(part_start_s + download_duration_seconds, total_duration)
            if part_end_s > part_start_s + 0.1:
                part_start_s_rounded = round(part_start_s, 2)
                part_end_s_rounded = round(part_end_s, 2)
                unique_download_chunks.add((part_start_s_rounded, part_end_s_rounded))
            current_part_start += download_duration_seconds
    check_cancellation(session_id)
    if not unique_download_chunks:
        logger.warning("No download chunks identified as necessary based on analysis blocks.")
        return True
    logger.info(f"Identified {len(unique_download_chunks)} unique {DOWNLOAD_DURATION_MINUTES}-minute download chunks required.")

    MAX_CONCURRENT_DOWNLOADS = 3
    download_pool = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS, thread_name_prefix='Downloader_UniqueChunk')
    download_futures: Dict[concurrent.futures.Future, tuple[float, float, str]] = {}
    download_results: Dict[tuple[float, float], Dict[str, Any]] = {}
    temp_files_to_clean: List[Path] = []
    logger.info("Submitting unique download tasks...")
    for chunk_start, chunk_end in sorted(list(unique_download_chunks)):
        check_cancellation(session_id)
        part_fname = f"unique_part_{format_time(chunk_start)}_to_{format_time(chunk_end)}".replace(":", "_").replace(" ", "")
        logger.debug(f"Queueing download: {part_fname} ({chunk_start:.2f}s - {chunk_end:.2f}s)")
        future = download_pool.submit(download_segment, youtube_url, chunk_start, chunk_end, part_fname, session_video_dir, session_id)
        download_futures[future] = (chunk_start, chunk_end, part_fname)
        download_results[(chunk_start, chunk_end)] = {"status": "pending", "path": None}
    check_cancellation(session_id)
    logger.info(f"Waiting for {len(download_futures)} unique download tasks to complete...")
    download_success_count = 0
    download_fail_count = 0
    for future in concurrent.futures.as_completed(download_futures):
        check_cancellation(session_id)
        chunk_start, chunk_end, part_fname = download_futures[future]
        result_key = (chunk_start, chunk_end)
        try:
            segment_path_str = future.result()
            check_cancellation(session_id)
            if segment_path_str and Path(segment_path_str).exists() and Path(segment_path_str).stat().st_size > 100:
                logger.info(f"✅ Download success: {part_fname}")
                segment_path = Path(segment_path_str)
                download_results[result_key]["status"] = "success"
                download_results[result_key]["path"] = segment_path
                temp_files_to_clean.append(segment_path)
                download_success_count += 1
            else:
                logger.error(f"❌ Download failed (no path returned or file invalid): {part_fname}")
                download_results[result_key]["status"] = "failed"
                download_fail_count += 1
        except Exception as exc:
            logger.error(f"❌ Download exception for chunk {part_fname} ({chunk_start}-{chunk_end}): {exc}", exc_info=False)
            download_results[result_key]["status"] = "failed"
            download_fail_count += 1
    download_pool.shutdown(wait=True)
    check_cancellation(session_id)
    logger.info(f"Unique Downloads Complete: {download_success_count} Success, {download_fail_count} Failed.")
    if download_success_count == 0 and len(unique_download_chunks) > 0:
        logger.error("All unique chunk downloads failed. Cannot proceed with analysis.")
        return False

    overall_success = True
    logger.info(f"\n--- Assembling and Analyzing {len(analysis_segments_times)} analysis blocks ---")
    try:
        check_cancellation(session_id)
        with open(combined_analysis_file, 'a', encoding='utf-8') as outfile:
            check_cancellation(session_id)
            if outfile.tell() == 0:
                logger.warning(f"{combined_analysis_file.name} was empty. Writing initial header.")
                outfile.write("## Combined Analysis (Code Extraction Follows)\n\n")
            outfile.write("\n\n---\n## Video Code Extraction Analysis\n---\n\n")
            check_cancellation(session_id)
            for analysis_idx, times in enumerate(analysis_segments_times):
                check_cancellation(session_id)
                analysis_start_s = times['start']
                analysis_end_s = times['end']
                block_duration = analysis_end_s - analysis_start_s
                logger.info(f"\n--- Processing Analysis Block {analysis_idx+1}/{len(analysis_segments_times)} ({format_time(analysis_start_s)} - {format_time(analysis_end_s)}) ---")
                required_parts_for_block: List[Path] = []
                block_parts_available = True
                current_part_start_s = math.floor(analysis_start_s / download_duration_seconds) * download_duration_seconds
                logger.debug(f"Finding parts for block {analysis_start_s:.2f}s - {analysis_end_s:.2f}s")
                while current_part_start_s < analysis_end_s and current_part_start_s < total_duration:
                    check_cancellation(session_id)
                    part_start_s = current_part_start_s
                    part_end_s = min(part_start_s + download_duration_seconds, total_duration)
                    part_key = (round(part_start_s, 2), round(part_end_s, 2))
                    if part_end_s <= part_start_s + 0.1:
                        current_part_start_s += download_duration_seconds
                        continue
                    logger.debug(f"  Checking for part key: {part_key} ({format_time(part_start_s)}-{format_time(part_end_s)})")
                    if part_key in download_results and download_results[part_key]["status"] == "success":
                        part_path = download_results[part_key]["path"]
                        if part_path and part_path.exists():
                            required_parts_for_block.append(part_path)
                            logger.debug(f"  Found required part: {part_path.name}")
                        else:
                            logger.error(f"  Required part file MISSING for key {part_key}: {part_path}. Block cannot be analyzed.")
                            block_parts_available = False
                            break
                    else:
                        logger.error(f"  Required part for key {part_key} (covering {format_time(part_start_s)}-{format_time(part_end_s)}) was NOT downloaded successfully. Block cannot be analyzed.")
                        block_parts_available = False
                        break
                    current_part_start_s += download_duration_seconds
                check_cancellation(session_id)
                if not block_parts_available:
                    logger.error(f"Skipping analysis for block {analysis_idx+1} due to missing/failed download parts.")
                    outfile.write(f"## Segment Analysis ({format_time(analysis_start_s)} - {format_time(analysis_end_s)})\n\n[ERROR: Required video parts missing or failed download for this time range.]\n\n---\n\n")
                    overall_success = False
                    continue
                if not required_parts_for_block:
                    logger.warning(f"No downloaded parts identified as needed for block {analysis_idx+1}, but block was marked available? Skipping analysis. Block range: {analysis_start_s}-{analysis_end_s}")
                    continue
                combined_block_path: Optional[Path] = None
                if len(required_parts_for_block) == 1:
                    combined_block_path = required_parts_for_block[0]
                    logger.info(f"Single part needed ({combined_block_path.name}), using directly for analysis.")
                else:
                    def get_start_time_from_name(p: Path) -> float:
                        match = re.search(r'_(\d{2}_\d{2}_\d{2})_to_', p.name)
                        if match:
                            try:
                                return parse_relative_timestamp(match.group(1).replace('_', ':'))
                            except:
                                return float('inf')
                        logger.warning(f"Could not parse start time from filename: {p.name}")
                        return float('inf')
                    check_cancellation(session_id)
                    required_parts_for_block.sort(key=get_start_time_from_name)
                    logger.debug(f"Parts to concatenate in order: {[p.name for p in required_parts_for_block]}")
                    concat_filename = f"analysis_block_{analysis_idx+1}_{format_time(analysis_start_s)}_to_{format_time(analysis_end_s)}.webm".replace(":", "_").replace(" ", "")
                    concat_output_path = session_video_dir / concat_filename
                    logger.info(f"Concatenating {len(required_parts_for_block)} parts into {concat_output_path.name}...")
                    check_cancellation(session_id)
                    concat_success = concatenate_segments([str(p) for p in required_parts_for_block], str(concat_output_path), session_temp_dir, session_id)
                    check_cancellation(session_id)
                    if concat_success:
                        combined_block_path = concat_output_path
                        temp_files_to_clean.append(combined_block_path)
                        logger.info("✅ Concatenation successful.")
                    else:
                        logger.error(f"❌ Concatenation failed for block {analysis_idx+1}.")
                        outfile.write(f"## Segment Analysis ({format_time(analysis_start_s)} - {format_time(analysis_end_s)})\n\n[ERROR: Concatenation failed for this segment.]\n\n---\n\n")
                        overall_success = False
                        continue
                check_cancellation(session_id)
                if not combined_block_path or not combined_block_path.exists():
                    logger.error(f"No valid video file available (missing or failed concat) for analysis block {analysis_idx+1}.")
                    outfile.write(f"## Segment Analysis ({format_time(analysis_start_s)} - {format_time(analysis_end_s)})\n\n[ERROR: Video file for analysis is missing or invalid.]\n\n---\n\n")
                    overall_success = False
                    continue
                logger.info(f"Analyzing video block file: {combined_block_path.name}")
                check_cancellation(session_id)
                analysis_result_text = analyze_combined_segment(
                    str(combined_block_path), analysis_start_s, analysis_end_s, _multimodal_llm, session_id
                )
                check_cancellation(session_id)
                if analysis_result_text is None or "[ERROR" in analysis_result_text:
                    logger.error(f"Analysis failed for block {analysis_idx+1}.")
                    outfile.write(analysis_result_text or f"## Segment Analysis ({format_time(analysis_start_s)} - {format_time(analysis_end_s)})\n\n[ERROR: Analysis failed or returned empty for this segment.]\n\n---\n\n")
                    overall_success = False
                else:
                    logger.info(f"Analysis successful for block {analysis_idx+1}. Appending to file.")
                    outfile.write(analysis_result_text)
                    outfile.write("\n\n---\n\n")
                    outfile.flush()
                check_cancellation(session_id)
            logger.info("Finished processing all analysis blocks.")
    except CancelledError:
        logger.warning(f"Code Extraction cancelled for session {session_id}.")
        raise
    except Exception as e:
        logger.exception(f"An error occurred during the analysis block processing loop: {e}")
        overall_success = False
    finally:
        check_cancellation(session_id)
        logger.info(f"Final cleanup of {len(temp_files_to_clean)} temporary video file(s)...")
        cleaned_count = 0
        for f_path in temp_files_to_clean:
            try:
                check_cancellation(session_id)
                if f_path and f_path.exists():
                    f_path.unlink()
                    logger.debug(f"Deleted temp video file: {f_path.name}")
                    cleaned_count += 1
            except Exception as clean_e:
                logger.warning(f"Could not delete temp video file {f_path}: {clean_e}")
        logger.info(f"Cleaned up {cleaned_count} video files.")
        try:
            check_cancellation(session_id)
            if session_video_dir.exists() and not any(session_video_dir.iterdir()):
                session_video_dir.rmdir()
                logger.info(f"Removed empty directory: {session_video_dir}")
        except Exception as rmdir_e:
            logger.warning(f"Could not remove empty video segment dir {session_video_dir}: {rmdir_e}")
    return overall_success

# --- Main Tool Function ---
def run_code_extraction(
    youtube_url: str,
    combined_analysis_file_path: str,
    session_id: str,
    time_ranges: Optional[List[str]] = None
) -> dict:
    logger.info(f"Starting Code Extraction for URL: {youtube_url} (Session: {session_id})")
    logger.info(f"Appending to: {combined_analysis_file_path}")
    if time_ranges:
        logger.info(f"Targeted ranges: {time_ranges}")
    
    try:
        logger.debug(f"CHECKPOINT CE_RCE1: Start run_code_extraction for {session_id}")
        check_cancellation(session_id)
        if not _initialize_multimodal_llm():
            return {"status": "error", "message": "Failed to initialize Multimodal LLM."}
        check_cancellation(session_id)
        analysis_file = Path(combined_analysis_file_path)
        if not analysis_file.exists():
            logger.error(f"File not found: {combined_analysis_file_path}")
            return {"status": "error", "message": f"Input analysis file not found: {combined_analysis_file_path}"}
        check_cancellation(session_id)
        normalized_url = normalize_youtube_url(youtube_url, session_id)
        check_cancellation(session_id)
        if not normalized_url:
            return {"status": "error", "message": "Invalid YouTube URL provided."}
        logger.debug(f"CHECKPOINT CE_RCE2: Before process_video_for_code for {session_id}")
        check_cancellation(session_id)
        success = process_video_for_code(normalized_url, analysis_file, session_id, time_ranges)
        logger.debug(f"CHECKPOINT CE_RCE3: After process_video_for_code for {session_id}")
        check_cancellation(session_id)
        if success:
            logger.info(f"Extraction completed successfully for session {session_id}.")
            return {"status": "success", "message": "Code extraction analysis appended successfully."}
        logger.warning(f"Extraction completed with failures for session {session_id}.")
        return {"status": "success", "message": "Extraction finished, but some segments may have failed. Check logs and file."}
    
    except CancelledError as ce:
        logger.warning(f"Code Extraction cancelled for session {session_id}: {ce}")
        return {"status": "cancelled", "message": "Processing cancelled by user."}
    except Exception as e:
        logger.exception(f"Critical error for session {session_id}: {e}")
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}
    finally:
        pass