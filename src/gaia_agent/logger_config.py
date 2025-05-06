import sys
import os
from pathlib import Path
from loguru import logger

# --- Configuration ---
LOG_LEVEL_CONSOLE = "DEBUG"  # Level for console output (e.g., DEBUG, INFO, WARNING)
LOG_LEVEL_FILE = "INFO"  # Level for file output
LOG_DIR = Path("logs")  # Directory to store log files (relative to project root)
LOG_FILENAME = "gaia_agent.log"  # Log file name
LOG_ROTATION = "10 MB"  # Rotate log file when it reaches 10 MB
LOG_RETENTION = "7 days"  # Keep logs for 7 days
LOG_COMPRESSION = "zip"  # Compress rotated logs
LOG_FORMAT_CONSOLE = (  # Detailed format for console with colors
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
LOG_FORMAT_FILE = (  # Simpler format for file
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} - {message}"
)
# --- End Configuration ---

# Ensure log directory exists
try:
    # Try finding project root assuming this file is in src/gaia_agent
    project_root = Path(__file__).parent.parent.parent
    log_path = project_root / LOG_DIR
    log_path.mkdir(parents=True, exist_ok=True)
    log_file_path = log_path / LOG_FILENAME
except Exception as e:
    print(f"Error creating log directory: {e}. Logs might not be saved to file.")
    log_file_path = LOG_FILENAME  # Fallback to current dir if path fails

# Remove default handler to prevent duplicate messages in console
logger.remove()

# Configure Console Logger
logger.add(
    sys.stderr,  # Output to standard error
    level=LOG_LEVEL_CONSOLE,
    format=LOG_FORMAT_CONSOLE,
    colorize=True,
    enqueue=True,  # Make console logging non-blocking
)

# Configure File Logger
logger.add(
    log_file_path,
    level=LOG_LEVEL_FILE,
    format=LOG_FORMAT_FILE,
    rotation=LOG_ROTATION,
    retention=LOG_RETENTION,
    compression=LOG_COMPRESSION,
    encoding="utf-8",
    enqueue=True,  # Asynchronous logging
    backtrace=True,  # Log full tracebacks for exceptions
    diagnose=False,  # Set to True for more detailed diagnostics if needed
)

logger.info(
    f"Logger initialized. Console Level: {LOG_LEVEL_CONSOLE}, File Level: {LOG_LEVEL_FILE}, File Path: {log_file_path}"
)

if __name__ == "__main__":
    # Test logging
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
