# backend/tools/tool_guide_generator.py
import os
import sys
import logging
import pathlib
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import google.generativeai as genai

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Import Cancellation Helpers ---
try:
    from utils.cancellation_utils import check_cancellation, CancelledError
    CANCELLATION_ENABLED = True
except ImportError:
    tool_name = __name__.split('.')[-1]  # Get tool name for logging
    logger.error(f"Could not import cancellation checks from utils. Cancellation disabled for {tool_name}.")
    class CancelledError(Exception): pass
    def check_cancellation(session_id: str): pass
    CANCELLATION_ENABLED = False

# --- Import LLM Utils ---
try:
    from utils.llm_utils import call_gemini_api_stream_with_retries, call_gemini_with_retries
except ImportError:
    logger.warning("Could not import retry utils relatively, trying direct.")
    from llm_utils import call_gemini_api_stream_with_retries, call_gemini_with_retries

# --- Configuration ---
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / '.env')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GUIDE_MODEL_NAME = os.getenv("GUIDE_MODEL", "gemini-2.5-pro-exp-03-25")

# --- Directory Setup ---
TEMP_BASE_DIR = pathlib.Path(os.getenv("TEMP_DATA_DIR", "./temp_youtube_data")).resolve()

# --- Pydantic Input Schema ---
class GuideGenerationArgs(BaseModel):
    combined_analysis_file_path: str = Field(description="The absolute path to the completed combined_analysis.md file containing metadata, transcript, and code extraction results.")
    session_id: str = Field(description="A unique identifier for the current processing session.")
    user_guide_instructions: Optional[str] = Field(None, description="Optional specific instructions from the user to guide the final Markdown generation.")

# --- Global Variables for Initialized Clients ---
_guide_genai_configured = False
_guide_generation_model = None

def _initialize_guide_gemini() -> bool:
    """Initializes the Gemini client for guide generation."""
    global _guide_genai_configured, _guide_generation_model
    if _guide_genai_configured:
        return True
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not found for guide generation.")
        return False
    try:
        logger.info(f"Initializing Guide Generation Model ({GUIDE_MODEL_NAME})...")
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        generation_config = genai.GenerationConfig(
            temperature=0.4,
        )
        _guide_generation_model = genai.GenerativeModel(
            GUIDE_MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        logger.info(f"Guide Generation Model ({GUIDE_MODEL_NAME}) Initialized.")
        _guide_genai_configured = True
        return True
    except Exception as e:
        logger.exception(f"Error initializing guide generation model ({GUIDE_MODEL_NAME}): {e}")
        _guide_genai_configured = False
        return False

# --- Helper Functions ---

def read_input_file(file_path: pathlib.Path, session_id: str) -> str:
    """Reads the content of the input analysis file with cancellation checks."""
    check_cancellation(session_id)
    logger.info(f"Reading input file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Successfully read {len(content)} characters from {file_path.name}")
        check_cancellation(session_id)
        return content
    except FileNotFoundError:
        logger.error(f"Input analysis file not found at '{file_path}'")
        raise
    except Exception as e:
        logger.exception(f"Error reading input file '{file_path}': {e}")
        raise

def construct_prompt(analysis_content: str, user_guide_instructions: Optional[str]) -> str:
    """Creates the detailed prompt for the Gemini API call, expecting transcripts and using user guidance."""
    logger.info("Constructing prompt for guide generation...")
    user_guide_section = f"""
**User Guidance for Final Guide Generation:**
Please focus on the following aspects when generating the final guide:
---
{user_guide_instructions if user_guide_instructions else "Standard comprehensive guide generation."}
---
"""
    prompt = f"""
    You are an expert technical writer and full-stack developer creating a final, runnable, step-by-step project guide from a **structured analysis file containing both video segment analysis and a full audio transcript**. Your specific constraint is that **all resulting code must use Google Gemini models**.

    **Task:** Convert the following **structured analysis text** into a single, polished, and comprehensive Markdown document. This document must guide a developer to replicate the *specific project described in the analysis*. Synthesize information across segments, **leverage the full transcript for context and accuracy**, assemble complete files accurately, and perform necessary code modifications. Follow any specific user guidance provided.

    {user_guide_section}

    **Input Analysis (Structured - Includes Transcript):**
    The analysis below contains:
    *   Segment-by-segment analysis (`## Segment Analysis...`) based on video frames.
    *   A final `## Full Video Transcript` section.
    *   Within segments: Setup steps, Dependencies, Code Snippets (with File Path Guesses, Scope, and Purpose comments), and Key Concepts.

    ```text
    {analysis_content}
    ```

    **Output Requirements:**

    1.  **Format:** Markdown (`.md`).
    2.  **Focus:** Strictly base the guide on the **Input Analysis (video segments AND full transcript)** and **User Guidance**. Setup should include cloning the repo and installing dependencies. The guide should be runnable and comprehensive, covering all necessary steps to replicate the project.
    3.  **Structure:**
        *   Logical sections (`##`, `###`) relevant to the project in the analysis (e.g., Setup, Core Logic, API if applicable, Testing, Deployment).
        *   Numbered steps, logically ordered.
        *   Aggregate Information: Combine related setup steps/dependencies across segments. Infer package scope if possible and relevant.
    4.  **Code Generation & Assembly:**
        *   Use `# File Path Guess:` to group snippets.
        *   Synthesize **all** snippets for a file into **one complete code block**. Use `// Purpose:` and the **Full Video Transcript** for context, ordering, and adding necessary boilerplate/imports/exports. Resolve potential conflicts between snippets for the same file logically.
        *   Label final code blocks clearly (e.g., ```python:src/main.py```).
    5.  **Content:**
        *   Clear explanations, enriched by **referencing the transcript** for intent/details.
        *   Aggregate dependencies mentioned visually *or* verbally (in transcript). Ensure Gemini dependencies are present and correct; convert/exclude others.
        *   Include setup commands mentioned visually *or* verbally.
        *   Include SQL schema if present, ensuring `vector(768)` for Gemini embeddings.
        *   **CRITICAL: Model Conversion to Gemini:** Verify and Ensure **ALL** LLM/Embedding code uses Gemini models (`ChatGoogleGenerativeAI`, `GoogleGenerativeAIEmbeddings`), `GOOGLE_API_KEY`, and correct model names (e.g., `gemini-pro`, `embedding-001`). Handle vector dimensions (`vector(768)`).
        *   **CRITICAL: Use Transcript:** Actively use the `## Full Video Transcript` to verify steps, clarify code purpose, capture details mentioned only verbally, and ensure consistency between visual analysis and spoken word.
        *   Address TODOs/Gaps appropriately.
    6.  **Consistency Check (Internal):** Briefly review if generated file interactions (imports/calls) seem correct and if core components described in the analysis/transcript are present.
    7.  **Tone:** Professional, accurate, runnable, reflecting any user guidance.

    **Constraint:** Generate the *entire* Markdown guide in a single response.

    **Begin Generating the Final Step-by-Step Guide based ONLY on the Input Analysis (including transcripts) and User Guidance provided:**
    """
    logger.info(f"Prompt constructed (length: {len(prompt)} characters)")
    return prompt

# --- Main Tool Function ---
def run_guide_generation(
    combined_analysis_file_path: str,
    session_id: str,
    user_guide_instructions: Optional[str] = None
) -> dict:
    """
    Reads the combined analysis file, generates a refined Markdown guide using Gemini,
    and saves it to the session's temporary directory.
    Returns a dictionary with status and path to the final guide file.
    """
    logger.info(f"Starting Guide Generation for session: {session_id}")
    logger.info(f"Reading analysis from: {combined_analysis_file_path}")

    final_guide_file_path = None

    try:
        check_cancellation(session_id)
        if not _initialize_guide_gemini():
            return {"status": "error", "message": "Failed to initialize Guide Generation LLM.", "final_guide_path": None}

        analysis_file = pathlib.Path(combined_analysis_file_path)
        session_dir = TEMP_BASE_DIR / session_id
        final_guide_file_path = session_dir / f"final_guide_{session_id}.md"
        check_cancellation(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        check_cancellation(session_id)
        analysis_content = read_input_file(analysis_file, session_id)
        if not analysis_content.strip():
            logger.error("Input analysis file is empty.")
            return {"status": "error", "message": "Input analysis file is empty.", "final_guide_path": None}

        check_cancellation(session_id)
        prompt = construct_prompt(analysis_content, user_guide_instructions)

        check_cancellation(session_id)
        logger.info("Generating final guide via Gemini API (non-streaming)...")
        response = call_gemini_with_retries(
            model_input=prompt,
            model_instance=_guide_generation_model,
            model_name=GUIDE_MODEL_NAME
        )
        check_cancellation(session_id)

        # Process response
        generated_guide = None
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            logger.error(f"Guide generation blocked! Reason: {block_reason}")
            if hasattr(response.prompt_feedback, 'safety_ratings'):
                logger.error(f"Safety Ratings: {response.prompt_feedback.safety_ratings}")
            raise Exception(f"Guide generation blocked: {block_reason}")
        elif response.text:
            generated_guide = response.text
            logger.info(f"Guide generation successful. Length: {len(generated_guide)}")
        else:
            logger.warning("Guide generation call succeeded but returned no text content.")
            generated_guide = "[ERROR: Guide generation returned no text content]"

        if "[ERROR:" in generated_guide:
            logger.error("Guide generation failed (error marker found in response string).")
            try:
                with open(final_guide_file_path, 'w', encoding='utf-8') as f:
                    f.write(generated_guide)
                logger.info(f"Error guide content saved to: {final_guide_file_path}")
            except Exception:
                pass
            return {"status": "error", "message": "Guide generation failed (see saved file/logs).", "final_guide_path": str(final_guide_file_path)}

        check_cancellation(session_id)
        logger.info(f"Saving final guide to: {final_guide_file_path}")
        with open(final_guide_file_path, 'w', encoding='utf-8') as f:
            f.write(generated_guide)
        logger.info("Final guide saved successfully.")

        check_cancellation(session_id)
        return {
            "status": "success",
            "message": "Final guide generated successfully.",
            "final_guide_path": str(final_guide_file_path.resolve())
        }

    except CancelledError as ce:
        logger.warning(f"Guide Generation cancelled for session {session_id}: {ce}")
        return {"status": "cancelled", "message": "Processing cancelled by user.", "final_guide_path": None}
    except FileNotFoundError:
        logger.error(f"Combined analysis file not found at {combined_analysis_file_path}")
        return {"status": "error", "message": f"Input analysis file not found: {combined_analysis_file_path}", "final_guide_path": None}
    except Exception as e:
        logger.exception(f"An unexpected error occurred during guide generation for session {session_id}: {e}")
        return {"status": "error", "message": f"An unexpected error occurred: {e}", "final_guide_path": None}