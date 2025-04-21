# TubeCoder 

TubeCoder is a web application that takes a YouTube tutorial URL, analyzes its content using AI, and generates a corresponding code project structure and step-by-step guide. It also allows users to ask follow-up questions about the video content using Retrieval-Augmented Generation (RAG).

## Features

* **YouTube URL Processing:** Input a YouTube video URL to initiate analysis.
* **AI-Powered Analysis:**
    * Extracts video metadata (title, description, links).
    * Downloads and transcribes audio using Google Gemini.
    * Analyzes video segments using a multimodal Google Gemini model to extract code snippets and procedural steps.
* **RAG Indexing:** Creates a session-specific FAISS vector index from the transcript and metadata for contextual Q&A.
* **Guide Generation:** Synthesizes the analysis into a polished Markdown guide using Google Gemini.
* **Project Scaffolding:** Generates a downloadable project structure (.zip) with code files based on the final guide, using Google Gemini.
* **Conversational Q&A:** Ask follow-up questions about the processed video, answered using the RAG index.
* **Cancellation Support:** Allows attempting to stop long-running backend processes (works reliably in single-process environments).
* **Web Interface:** Simple chat-based UI built with Next.js and React.

## Architecture Overview

* **Frontend:** Next.js (React/TypeScript) application providing the user interface.
* **Backend:** Python application using FastAPI, Langchain (Agents & Tools), and Google Generative AI (Gemini) for processing and generation tasks. FAISS is used for vector storage.

## Tech Stack

**Backend:**
* Python 3.8+
* FastAPI
* Uvicorn
* Langchain (Agents, Google GenAI wrappers, FAISS, Document Loaders, Text Splitters)
* Google Generative AI SDK (`google-generativeai`)
* `python-dotenv`
* `yt-dlp` (for YouTube downloading)
* `pydub` (for audio manipulation - **Requires FFmpeg**)
* `faiss-cpu` (for vector indexing)

**Frontend:**
* Node.js 
* Next.js 14+
* React 18+
* TypeScript
* Tailwind CSS (implied)
* `uuid`
* `react-markdown` / `remark-gfm`
* Heroicons

**External Dependencies:**
* **FFmpeg:** Required by `pydub` for audio processing. **Must be installed separately** on the system running the backend and accessible in the system's PATH.

## Prerequisites

* **Python:** Version 3.8 or higher recommended.
* **Node.js:** A recent LTS version (e.g., 18.x, 20.x) and npm or yarn.
* **Git:** For cloning the repository.
* **FFmpeg:** Must be installed and accessible in your system's PATH for the backend audio processing to work. ([Download FFmpeg](https://ffmpeg.org/download.html))
* **Google Cloud / AI Studio API Key:** You need an API key enabled for the Google Generative AI (Gemini) API.

## Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/KingDavid2908/tube-coder.git
    cd tube-coder
    ```

2.  **Backend Setup:**
    * Navigate to the backend directory:
        ```bash
        cd backend
        ```
    * Create and activate a Python virtual environment:
        ```bash
        python -m venv venv
        # On Windows:
        .\venv\Scripts\activate
        # On macOS/Linux:
        source venv/bin/activate
        ```
    * Install Python dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    * **Create Environment File:** Copy the example environment file:
        ```bash
        # On Windows:
        copy .env.example .env
        # On macOS/Linux:
        cp .env.example .env
        ```
    * **Create `.env` file in the backend folder:** Open the `.env` file and add your Google Generative AI API key:
        ```dotenv
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"

        # Optional: You can customize model names if needed
        # AGENT_MODEL="gemini-2.5-flash-preview-04-17" 
        # MULTIMODAL_MODEL="gemini-2.5-flash-preview-04-17" 
        # GUIDE_MODEL="gemini-2.5-pro-exp-03-25"
        # FORMATTER_MODEL="gemini-2.5-pro-exp-03-25"
        # EMBEDDING_MODEL_NAME="models/embedding-001"
        # TEMP_DATA_DIR="./temp_youtube_data"
        # VECTORSTORE_DIR="./rag_indices"
        ```
    * **Verify FFmpeg:** Ensure `ffmpeg` is installed and accessible by running `ffmpeg -version` in your terminal. If not found, install it for your OS.

3.  **Frontend Setup:**
    * Navigate to the frontend directory (from the root `tube-coder` directory):
        ```bash
        cd ../frontend
        # Or just cd frontend if already in the root
        ```
    * Install Node.js dependencies:
        ```bash
        # Using npm
        npm install

        # Or using yarn
        # yarn install
        ```
    * **Create Environment File:** Copy the example environment file:
        ```bash
        # On Windows:
        copy .env.local.example .env.local
        # On macOS/Linux:
        cp .env.local.example .env.local
        ```
    * **Edit `.env.local`:** Open `.env.local` and configure the backend API URLs. For standard local setup, the defaults usually work:
        ```dotenv
        NEXT_PUBLIC_BACKEND_API_URL=http://localhost:8080
        NEXT_PUBLIC_BACKEND_CHAT_API_URL=http://localhost:8080/api/chat
        NEXT_PUBLIC_BACKEND_CANCEL_API_URL=http://localhost:8080/api/cancel_request
        NEXT_PUBLIC_BACKEND_DOWNLOAD_URL=http://localhost:8080/api/download_project
        ```
        *(Adjust the port if your backend runs on a different one)*

## Running the Application

1.  **Start the Backend Server:**
    * Open a terminal in the `backend` directory.
    * Activate the virtual environment (`.\venv\Scripts\activate` or `source venv/bin/activate`).
    * Run the FastAPI server:
        ```bash
        # For development with auto-reload (Cancellation works reliably):
        uvicorn main:app --host 0.0.0.0 --port 8080 --reload

        # OR for a single process without reload (Cancellation works reliably):
        # uvicorn main:app --host 0.0.0.0 --port 8080
        ```
    * Keep this terminal running.

2.  **Start the Frontend Server:**
    * Open a *separate* terminal in the `frontend` directory.
    * Run the Next.js development server:
        ```bash
        # Using npm
        npm run dev

        # Or using yarn
        # yarn dev
        ```
    * Keep this terminal running.

3.  **Access the Application:**
    * Open your web browser and navigate to `http://localhost:3000`.

## Note
The projects built are built to use gemini models since they are cheaper and more accesible for everyone, edit to use the model of your choice by editing the prompts.

## Usage

1.  **Process Video:** Paste a full YouTube video URL into the chat input and press Enter or click the send button. The backend will start the processing pipeline.
2.  **Ask Questions:** Once a video has been indexed (after the first step completes), you can ask follow-up questions about its content in the chat.
3.  **Stop Processing:** While the backend is working (indicated by the "Thinking..." state), you can click the "Stop" button to attempt cancellation. See the "Cancellation Feature" section below for details on reliability.
4.  **Download Project:** If the full pipeline completes successfully, a `[DOWNLOAD_PROJECT:...]` message will appear along with a "Download Project (.zip)" button. Click this button to download the generated code.

## Cancellation Feature

This application includes a feature to attempt stopping backend processing via the "Stop" button.

* **How it Works:** The button sends a request to the backend, which sets an internal, in-memory flag for your session. Backend tasks check this flag periodically and should stop if the flag is set.
* **Reliability:** This mechanism functions reliably when running the backend server as a **single process** (e.g., using `uvicorn main:app --reload` or `uvicorn main:app`).
* **Important Limitation:** If deploying the backend with **multiple worker processes** (e.g., `gunicorn -w 4 main:app`), this cancellation feature **may not work reliably** because the in-memory flag is not shared between processes.

## Environment Variables

### Backend (`backend/.env`)

* `GOOGLE_API_KEY` (**Required**): Your API key for Google Generative AI Studio / Google Cloud AI Platform.
* `AGENT_MODEL`: The Gemini model used for the main agent logic.
* `MULTIMODAL_MODEL`: The Gemini model used for video analysis.
* `GUIDE_MODEL`: The Gemini model used for generating the final Markdown guide.
* `FORMATTER_MODEL`: The Gemini model used for generating the project file structure.
* `EMBEDDING_MODEL_NAME`: The Google embedding model for RAG.
* `TEMP_DATA_DIR`: Directory to store temporary session data.
* `VECTORSTORE_DIR`: Directory to store FAISS vector indexes.
* `YT_DLP_PATH`: Path to the `yt-dlp` executable.
* `FFMPEG_PATH`: Path to the `ffmpeg` executable.

### Frontend (`frontend/.env.local`)

* `NEXT_PUBLIC_BACKEND_API_URL`: Base URL for the backend (default: `http://localhost:8080`).
* `NEXT_PUBLIC_BACKEND_CHAT_API_URL`: Full URL for the chat endpoint (default: `http://localhost:8080/api/chat`).
* `NEXT_PUBLIC_BACKEND_CANCEL_API_URL`: Full URL for the cancel endpoint (default: `http://localhost:8080/api/cancel_request`).
* `NEXT_PUBLIC_BACKEND_DOWNLOAD_URL`: Full URL for the download endpoint (default: `http://localhost:8080/api/download_project`).

## License

Apache License 2.0.