# backend/agent/tubecoder_agent.py
import os
import pathlib
import re
import logging
from datetime import datetime
from typing import Optional, Dict
from uuid import uuid4
from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool  

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TubeCoderAgent")

# --- Import Cancellation Utils ---
try:
    from utils.cancellation_utils import check_cancellation, CancelledError
    CANCELLATION_ENABLED = True
except ImportError:
    logger.error("Could not import cancellation checks from utils. Cancellation checks disabled in agent.")
    class CancelledError(Exception): pass
    def check_cancellation(session_id: str): pass
    CANCELLATION_ENABLED = False

# Langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool, StructuredTool
from langchain.memory import ConversationBufferWindowMemory
from pydantic import BaseModel, Field
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    logger.error("FAISS not found. Install with 'pip install faiss-cpu langchain-community'.")
    class FAISS: pass
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pathlib import Path

# Import tool functions and args schemas
try:
    from ..tools.tool_youtube_processor import run_youtube_indexing, YouTubeIndexArgs
    from ..tools.tool_code_extractor import run_code_extraction, CodeExtractionArgs
    from ..tools.tool_guide_generator import run_guide_generation, GuideGenerationArgs
    from ..tools.tool_code_formatter import run_project_creation, ProjectCreationArgs
except ImportError:
    logging.warning("Could not use relative imports for tools, trying direct.")
    from tools.tool_youtube_processor import run_youtube_indexing, YouTubeIndexArgs
    from tools.tool_code_extractor import run_code_extraction, CodeExtractionArgs
    from tools.tool_guide_generator import run_guide_generation, GuideGenerationArgs
    from tools.tool_code_formatter import run_project_creation, ProjectCreationArgs

# --- Configuration ---
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / '.env')
AGENT_MODEL_NAME = os.getenv("AGENT_MODEL", "gemini-2.5-flash-preview-04-17")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Get Base Temp Dir ---
TEMP_BASE_DIR = pathlib.Path(os.getenv("TEMP_DATA_DIR", "./temp_youtube_data")).resolve()
VECTORSTORE_BASE_PATH = pathlib.Path(os.getenv("VECTORSTORE_DIR", "./rag_indices")).resolve()

# --- Tool Definitions ---
tools = [
    StructuredTool.from_function(
        func=run_youtube_indexing,
        name="YouTubeContentIndexer",
        description="Processes a YouTube URL to download audio, get metadata/transcript, save it to a session file, and create/update a session-specific RAG vector index. MUST be called first for any new YouTube URL. Input requires 'youtube_url' and 'session_id'. Returns a dictionary with status and 'combined_file_path'.",
        args_schema=YouTubeIndexArgs,
    ),
    StructuredTool.from_function(
        func=run_code_extraction,
        name="VideoCodeExtractor",
        description="Analyzes video segments (full video or specific 'time_ranges') for code/steps using multimodal AI. Appends structured analysis results to the markdown file specified by 'combined_analysis_file_path'. Input requires 'youtube_url', 'combined_analysis_file_path', 'session_id', and optional 'time_ranges' list (e.g., ['1:30-2:45']). Returns a dictionary with status.",
        args_schema=CodeExtractionArgs,
    ),
    StructuredTool.from_function(
        func=run_guide_generation,
        name="GuideRefiner",
        description="Generates a final, polished Markdown guide from a completed combined analysis file. Input requires 'combined_analysis_file_path', 'session_id', and optional 'user_guide_instructions'. Returns a dictionary with status and 'final_guide_path'.",
        args_schema=GuideGenerationArgs,
    ),
    StructuredTool.from_function(
        func=run_project_creation,
        name="ProjectStructureGenerator",
        description="Parses a final Markdown guide, creates the project file structure and code in a temporary server directory based on the 'session_id'. Input requires 'final_guide_path' and 'session_id'. Returns a dictionary with status, 'project_directory_path' (the temporary dir created), and 'setup_file_path' (path to setup instructions within that dir).",
        args_schema=ProjectCreationArgs,
    ),
]

# --- Agent Prompt Template ---
SYSTEM_PROMPT = """You are TubeCoder, an AI assistant that processes YouTube tutorial videos to generate code projects and answers questions about the video content.
**Workflow Rules:**
**Initial Request:** If the user provides a YouTube URL, your primary goal is to execute the full code generation pipeline using the available tools in sequence: YouTubeContentIndexer -> VideoCodeExtractor -> GuideRefiner -> ProjectStructureGenerator. **Use the `session_id` provided in the input for all tool calls.** If no `session_id` was provided, you MUST generate one and use it consistently. Respond ONLY with the final success message containing [DOWNLOAD_PROJECT:{session_id}] (using the correct session ID you used) or an error message if a tool fails.
**Follow-up Questions:** If the user asks a question *after* a video has been processed (i.e., the input doesn't look like a new URL processing request), you MUST use the provided chat history ({chat_history}) and the retrieved context ({retrieved_context}) from the video's transcript/metadata to answer the question directly. Do NOT call the code generation tools for these questions unless specifically asked to regenerate something. If no relevant context is found or the video hasn't been processed, state that you cannot answer based on the video content.
**Tool Usage:** Only use tools when processing an initial YouTube URL request. For follow-up questions, answer directly based on history and retrieved context.

**IMPORTANT:** When answering follow-up questions, base your answer *only* on the chat history and the following retrieved context. If the context doesn't contain the answer, say so.
**Retrieved Context:**
{retrieved_context}
**NOTE:** The user might not explicitly state that it is a youtube url. You should be able to detect it from the input. If you are not sure, ask the user for clarification.
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# --- Agent Initialization ---
llm = None
agent = None
agent_executor = None

# In-memory store (replace for production)
session_memory: Dict[str, ConversationBufferWindowMemory] = {}
_rag_embeddings_model = None

def _initialize_rag_embeddings():
    global _rag_embeddings_model
    if _rag_embeddings_model:
        return True
    if not GOOGLE_API_KEY:
        logger.error("API key missing for RAG embeddings")
        return False
    try:
        EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "models/embedding-001")
        _rag_embeddings_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY
        )
        logger.info("RAG Embeddings Initialized.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG embeddings: {e}")
        return False

def get_agent_executor():
    """Initializes and returns the agent executor (stateless regarding memory)."""
    global llm, agent, agent_executor
    if agent_executor:
        return agent_executor

    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not configured for agent.")
        raise ValueError("GOOGLE_API_KEY is not set.")

    try:
        llm = ChatGoogleGenerativeAI(
            model=AGENT_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            convert_system_message_to_human=True,
        )
        agent = create_tool_calling_agent(llm, tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors="Check your output and make sure it conforms!",
            max_iterations=15,
        ).with_config({"run_name": "TubeCoderAgentExecutor"})
        logger.info("Stateless Agent Executor created successfully.")
        return agent_executor
    except Exception as e:
        logger.exception(f"Failed to create agent executor: {e}")
        raise

async def handle_agent_request(user_input: str, session_id: Optional[str] = None) -> tuple[str, str]:
    """Handles a request, manages sessions externally, performs RAG context retrieval,
       and invokes the main agent executor in a threadpool."""
    generated_session_id = None
    current_session_id = session_id
    final_reply = ""

    try:
        # --- Session ID & Memory Management ---
        if not current_session_id:
            current_session_id = f"session_{uuid4()}"
            generated_session_id = current_session_id
            logger.info(f"Starting new session: {current_session_id}")

        if current_session_id not in session_memory:
            session_memory[current_session_id] = ConversationBufferWindowMemory(
                k=10, memory_key="chat_history", input_key="input", output_key="output", return_messages=True
            )
            logger.info(f"Created new memory for session: {current_session_id}")
            if not generated_session_id: generated_session_id = current_session_id

        current_memory_instance = session_memory[current_session_id]
        memory_variables = current_memory_instance.load_memory_variables({})
        chat_history = memory_variables.get("chat_history", [])

        # --- Determine Request Type ---
        is_url_processing_request = False
        url_pattern = r'(https?://[^\s<>"]+|www\.[^\s<>"]+)'
        is_youtube_link = bool(re.search(url_pattern, user_input) and ("youtube.com" in user_input or "youtu.be" in user_input))

        if is_youtube_link and \
           (len(chat_history) == 0 or "process" in user_input.lower() or "extract" in user_input.lower() or "generate" in user_input.lower() or "summarize this video" in user_input.lower()):
            is_url_processing_request = True
            logger.info(f"Detected URL processing request for session {current_session_id}.")
        else:
            logger.info(f"Detected follow-up question/input for session {current_session_id}.")

        # --- Prepare Context ---
        retrieved_context = "[No RAG context needed for tool execution]"

        if not is_url_processing_request:
            logger.info(f"Attempting RAG for Q&A (Session: {current_session_id})...")
            if not _initialize_rag_embeddings() or not _rag_embeddings_model:
                logger.error("Cannot perform RAG: Embeddings not initialized.")
                retrieved_context = "[RAG ERROR: Embeddings not available]"
            else:
                vectorstore_path = VECTORSTORE_BASE_PATH / current_session_id / "faiss_index"
                if vectorstore_path.exists():
                    try:
                        logger.info(f"Loading vector store: {vectorstore_path}")
                        vectorstore = FAISS.load_local(
                            folder_path=str(vectorstore_path),
                            embeddings=_rag_embeddings_model,
                            allow_dangerous_deserialization=True
                        )
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                        logger.info(f"Retrieving context for question: '{user_input}'")
                        docs = await run_in_threadpool(retriever.invoke, user_input)  
                        if docs:
                            retrieved_context = "\n\n".join([f"- {doc.page_content}" for doc in docs])
                            logger.info(f"Retrieved {len(docs)} context documents.")
                        else:
                            logger.info("No relevant documents found by retriever.")
                            retrieved_context = "[No relevant context found in the indexed video content for this question]"
                    except Exception as e:
                        logger.error(f"Failed to load or query vector store for session {current_session_id}: {e}")
                        retrieved_context = f"[RAG ERROR: Failed to load/query index: {e}]"
                else:
                    logger.warning(f"Vector store not found for session {current_session_id} at {vectorstore_path}. Cannot answer from video context.")
                    retrieved_context = "[RAG INFO: No video content has been indexed for this session yet.]"

        # --- Main Execution Block ---
        try:
            check_cancellation(current_session_id)

            logger.info(f"Invoking agent executor (Session: {current_session_id}). URL Processing Mode: {is_url_processing_request}")
            executor = get_agent_executor()
            agent_input_dict = {
                "input": user_input,
                "chat_history": chat_history,
                "session_id": current_session_id,
                "retrieved_context": retrieved_context,
            }
            logger.info(f"Agent input session_id before invoke: {agent_input_dict['session_id']}")

            response = await run_in_threadpool(executor.invoke, agent_input_dict)  

            check_cancellation(current_session_id)

            final_reply = response.get("output", "Agent did not produce an output.")

            logger.info(f"Agent raw output for session {current_session_id}: {final_reply}")
            if is_url_processing_request and "Project generation complete" in final_reply and "[DOWNLOAD_PROJECT:" not in final_reply:
                logger.warning("Agent output indicates success but missing download trigger! Adding it.")
                final_reply += f" [DOWNLOAD_PROJECT:{current_session_id}]"
            elif "[DOWNLOAD_PROJECT:{session_id}]" in final_reply:
                final_reply = final_reply.replace("{session_id}", current_session_id)

            try:
                current_memory_instance.save_context({"input": user_input}, {"output": final_reply})
                logger.info(f"Saved context to memory for session {current_session_id}")
            except Exception as mem_save_err:
                logger.error(f"Failed to save context to memory: {mem_save_err}")

            final_session_id = generated_session_id or current_session_id
            return final_reply, final_session_id

        except CancelledError as ce:
            logger.warning(f"Caught cancellation in handle_agent_request for session {current_session_id}: {ce}")
            return "[Processing stopped by user]", current_session_id
        except Exception as e:
            logger.exception(f"Critical error in handle_agent_request for session {current_session_id}: {e}")
            return f"Sorry, a critical error occurred: {e}", current_session_id

    except Exception as e:
        logger.exception(f"Critical error in handle_agent_request for session {current_session_id}: {e}")
        error_reply = f"Sorry, a critical error occurred: {e}"
        final_session_id = current_session_id or f"error_session_{uuid4()}"
        return error_reply, final_session_id