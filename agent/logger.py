# agent/logger.py
import logging
import os
from typing import Any, Dict

_logger = None

def get_logger(name: str = "agent") -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(ch)
    _logger = logger
    return logger

def log_event(event: str, payload: Dict[str, Any] | None = None) -> None:
    logger = get_logger()
    if payload:
        logger.info("%s | %s", event, payload)
    else:
        logger.info("%s", event)