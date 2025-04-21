# backend/utils/llm_utils.py
import time
import logging
import google.generativeai as genai
from google.api_core import exceptions
from typing import Any, List, Union, Sequence, Optional 

try:
    from langchain_core.messages import BaseMessage, AIMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    class BaseMessage: pass
    class AIMessage: pass
    class ChatGoogleGenerativeAI: pass


logger = logging.getLogger(__name__)

def call_gemini_api_stream_with_retries(
    prompt: str,
    model: genai.GenerativeModel,
    model_name: str
    ) -> str:
    """
    Calls the Gemini API using streaming, accumulates the response,
    and includes retry logic for transient ServiceUnavailable errors.
    """
    if not model: raise ValueError(f"{model_name} model instance not provided.")
    logger.info(f"Sending request to Model: {model_name} (Streaming w/ Retries)...")
    if "exp" in model_name: logger.warning(f"Note: '{model_name}' is an experimental model.")

    max_retries = 3
    base_delay_seconds = 2
    full_response = ""
    last_exception = None

    for attempt in range(max_retries):
        full_response = ""
        stream_interrupted = False
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to call {model_name}...")
            response_stream = model.generate_content(prompt, stream=True)
            logger.info(f"Receiving streamed response (Attempt {attempt + 1})...")
            chunk_count = 0
            for chunk in response_stream:
                block_reason = None; safety_ratings = None
                try:
                    if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback.block_reason: block_reason = chunk.prompt_feedback.block_reason; safety_ratings = getattr(chunk.prompt_feedback, 'safety_ratings', 'N/A')
                    elif hasattr(chunk, 'candidates') and chunk.candidates and hasattr(chunk.candidates[0], 'finish_reason') and chunk.candidates[0].finish_reason != 'STOP': block_reason = f"Candidate Finish Reason: {chunk.candidates[0].finish_reason}"; safety_ratings = getattr(chunk.candidates[0], 'safety_ratings', 'N/A')
                except Exception as feedback_err: logger.warning(f"Could not reliably check prompt feedback/safety: {feedback_err}")
                if block_reason: logger.error(f"Stream blocked by API on attempt {attempt + 1}. Reason: {block_reason}"); logger.error(f"Safety ratings: {safety_ratings}"); raise Exception(f"Stream blocked by API: {block_reason}")
                try:
                    chunk_text = getattr(chunk, 'text', '') 
                    if chunk_text:
                         full_response += chunk_text
                         chunk_count += 1
                except ValueError: logger.warning(f"Received chunk without text content (or end signal): {chunk}"); continue
                except Exception as chunk_err: logger.warning(f"Error processing chunk text: {chunk_err}. Chunk: {chunk}"); raise
            logger.info(f"Stream finished successfully on attempt {attempt + 1} after {chunk_count} chunks."); last_exception = None; break
        except exceptions.ServiceUnavailable as e: last_exception = e; logger.warning(f"Attempt {attempt + 1} failed: 503 Service Unavailable: {e}. Retrying after delay..."); stream_interrupted = True
        except Exception as e: last_exception = e; logger.exception(f"Attempt {attempt + 1} failed with unexpected error during streaming from {model_name}: {e}"); stream_interrupted = True; break 
        if stream_interrupted and attempt < max_retries - 1: delay = base_delay_seconds * (2 ** attempt); logger.info(f"Waiting {delay} seconds before next retry..."); time.sleep(delay)
        elif stream_interrupted: logger.error(f"Streaming failed after {attempt + 1} attempt(s). Last error: {last_exception}")
    if last_exception: error_msg = f"\n\n[API STREAM FAILED AFTER {attempt + 1} ATTEMPTS: {last_exception}]"; full_response += error_msg
    if not full_response.strip() and not last_exception: logger.error("Stream completed successfully but no text content was generated."); return "" 
    return full_response

def call_gemini_with_retries(
    model_input: Union[str, Sequence[Union[str, Any]]], 
    model_instance: Union[genai.GenerativeModel, Any], 
    model_name: str, 
    request_options: Optional[dict] = None 
    ) -> Any: 
    """
    Calls a Gemini model (native SDK or LangChain) with retries for non-streaming requests.
    Handles ServiceUnavailable errors.
    """
    if not model_instance:
        raise ValueError(f"{model_name} model instance not provided.")

    logger.info(f"Sending request to Model: {model_name} (Non-Streaming w/ Retries)...")
    if "exp" in model_name:
        logger.warning(f"Note: '{model_name}' is an experimental model.")

    max_retries = 3
    base_delay_seconds = 2
    last_exception = None

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to call {model_name}...")
            response = None
            if isinstance(model_instance, genai.GenerativeModel):
                response = model_instance.generate_content(
                    model_input,
                    request_options=request_options or {}
                )
                if response.prompt_feedback.block_reason:
                     logger.error(f"Call blocked by API. Reason: {response.prompt_feedback.block_reason}")
                     raise Exception(f"API Call Blocked: {response.prompt_feedback.block_reason}")

            elif LANGCHAIN_AVAILABLE and isinstance(model_instance, ChatGoogleGenerativeAI):
                if not isinstance(model_input, (str, list)):
                     raise TypeError(f"Invalid input type for LangChain invoke: {type(model_input)}. Expected str or list.")
                if isinstance(model_input, list) and not all(isinstance(m, BaseMessage) for m in model_input):
                     logger.debug("Input list doesn't contain BaseMessages, attempting simple conversion for invoke.")

                response = model_instance.invoke(model_input)
                metadata = getattr(response, 'response_metadata', {})
                if metadata.get('prompt_feedback', {}).get('block_reason'):
                     block_reason = metadata['prompt_feedback']['block_reason']
                     logger.error(f"Call blocked by API (via LangChain metadata). Reason: {block_reason}")
                     raise Exception(f"API Call Blocked (via LangChain metadata): {block_reason}")
            else:
                raise TypeError(f"Unsupported model instance type provided: {type(model_instance)}")

            logger.info(f"Call successful on attempt {attempt + 1}.")
            return response

        except exceptions.ServiceUnavailable as e:
            last_exception = e
            logger.warning(f"Attempt {attempt + 1} failed: 503 Service Unavailable: {e}. Retrying after delay...")
        except Exception as e: 
            last_exception = e
            logger.exception(f"Attempt {attempt + 1} failed with unexpected error during call to {model_name}: {e}")
            break

        if isinstance(last_exception, exceptions.ServiceUnavailable) and attempt < max_retries - 1:
            delay = base_delay_seconds * (2 ** attempt)
            logger.info(f"Waiting {delay} seconds before next retry...")
            time.sleep(delay)
        elif attempt >= max_retries - 1: 
            logger.error(f"Call failed after {max_retries} attempts. Last error: {last_exception}")
            break

    if last_exception:
        raise last_exception
    else:
        raise Exception("Gemini call failed after retries for an unknown reason.")