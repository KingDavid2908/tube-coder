# backend/utils/cancellation_utils.py
import logging
import threading
import subprocess

logger = logging.getLogger(__name__)

cancellation_flags = {}
active_processes = {} 
state_lock = threading.Lock()

# --- Custom Exception ---
class CancelledError(Exception):
    """Custom exception for cancellation."""
    pass

# --- Check Function ---
def check_cancellation(session_id: str):
    """Checks flag and raises CancelledError if set."""
    if not session_id:
        return

    with state_lock:
        session_state = cancellation_flags.get(session_id, {})
        is_cancelled = session_state.get('cancelled', False)
        logger.debug(f"CHECKING CANCELLATION for {session_id}: Flag value is {is_cancelled}")
        if is_cancelled:
            logger.warning(f"Cancellation DETECTED for session {session_id} during check.")
            raise CancelledError(f"Processing cancelled for session {session_id}")

# --- Helper to Initialize/Reset Flag ---
def initialize_flag(session_id: str):
    """Ensures flag dict exists and sets 'cancelled' to False."""
    if not session_id:
        return
    with state_lock:
        if session_id not in cancellation_flags:
            cancellation_flags[session_id] = {}
        cancellation_flags[session_id]['cancelled'] = False
        logger.debug(f"Initialized/Reset cancellation flag for session {session_id}")

# --- Helper to Set Flag (called from cancel endpoint) ---
def set_cancel_flag(session_id: str):
    """Sets the 'cancelled' flag to True for a session."""
    if not session_id:
        return
    with state_lock:
        if session_id not in cancellation_flags:
            cancellation_flags[session_id] = {}
        cancellation_flags[session_id]['cancelled'] = True
        logger.info(f"Cancellation flag SET for session {session_id}. Current flags: {repr(cancellation_flags)}")

# --- Helpers to Manage Active Processes (Keep for potential Popen use later) ---
def add_active_process(session_id: str, process: subprocess.Popen):
    if not session_id:
        return
    with state_lock:
        active_processes[session_id] = process
        logger.info(f"Stored process {process.pid} for session {session_id}")

def remove_active_process(session_id: str) -> subprocess.Popen | None:
    if not session_id:
        return None
    with state_lock:
        process = active_processes.pop(session_id, None)
        if process:
            logger.info(f"Removed process {process.pid} for session {session_id} from active list.")
        return process

def get_active_process(session_id: str) -> subprocess.Popen | None:
    if not session_id:
        return None
    with state_lock:
        return active_processes.get(session_id)