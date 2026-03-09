"""Logging configuration."""

import logging


def setup_logging(level="INFO", log_file=None):
    """Configure logging for the application."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

    return logging.getLogger("dataflow")
