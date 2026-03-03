# app_log.py
# Application-level logging for diagnostics and debugging

import logging
import sys
from pathlib import Path

LOG_FILENAME = "log.log"


def setupLogging(logDir=None):
    """
    Set up file + console logging. Call once from main.py.
    The log file appends across runs so testers dont lose anything.
    """
    if logDir:
        logPath = Path(logDir) / LOG_FILENAME
    else:
        logPath = Path(LOG_FILENAME)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # file handler - captures everything
    fileHandler = logging.FileHandler(logPath, encoding="utf-8")
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # console handler - only warnings and above so it doesnt spam the terminal
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.WARNING)
    consoleHandler.setFormatter(logging.Formatter(
        "[%(levelname)s] %(message)s"
    ))

    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)

    logging.info("Application started")
    logging.info(f"Python {sys.version}")
    logging.info(f"Platform: {sys.platform}")
    logging.info(f"Log file: {logPath.resolve()}")