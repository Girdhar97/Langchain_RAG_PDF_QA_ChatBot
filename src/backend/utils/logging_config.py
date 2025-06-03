import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = "logs"
LOG_FILE = "rag_backend.log"
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger():
    log_path = os.path.join(LOG_DIR, LOG_FILE)
    logger = logging.getLogger("rag_backend")
    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=5)
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(module)s.%(funcName)s: %(message)s'
    )
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger

# # -----Test the logging configuration--------- 

# if __name__ == "__main__":
#     # Test the logging configuration
#     logger = setup_logger()
    
#     logger.debug("This is a DEBUG message")
#     logger.info("This is an INFO message")
#     logger.warning("This is a WARNING message")
#     logger.error("This is an ERROR message")
#     logger.critical("This is a CRITICAL message")
    
#     print("\n✓ Logging configuration working correctly!")
