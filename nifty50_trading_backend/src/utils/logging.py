import logging
import os
from typing import Optional


_LOGGER_CONFIGURED = False


def _configure_root_logger(level: str) -> None:
    """Internal helper to configure the root logger once."""
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return
    level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _LOGGER_CONFIGURED = True


# PUBLIC_INTERFACE
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger configured using LOG_LEVEL env.

    LOG_LEVEL can be set in the environment or .env file and defaults to INFO.
    """
    env_level = os.getenv("LOG_LEVEL", "INFO")
    _configure_root_logger(env_level)
    return logging.getLogger(name if name else __name__)
