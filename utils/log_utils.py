import logging
import os

def setup_logging(log_file, fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s'):
    """
    Configure logging to write to the specified file under ./logs/.
    Creates the log directory if it does not exist.
    """
    log_path = log_file
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler(),
        ],
        force=True,
    )