"""Centralized logging configuration for the gplus-trees project."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.DEBUG,
    format_string: Optional[str] = None,
    handler_type: str = "stream"
) -> logging.Logger:
    """
    Set up centralized logging configuration for the project.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)
        handler_type: Type of handler - "stream", "file", or "both"
        
    Returns:
        Configured logger instance
    """
    # Default format if none provided
    if format_string is None:
        format_string = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    
    # Get root logger for the project
    logger = logging.getLogger("gplus_trees")
    
    # Avoid duplicate configuration
    if logger.hasHandlers():
        return logger
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Add stream handler
    if handler_type in ("stream", "both"):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    # Set level and prevent propagation to root logger
    logger.setLevel(level)
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Logger instance
    """
    # Ensure base logging is set up
    if not logging.getLogger("gplus_trees").hasHandlers():
        setup_logging()

    return logging.getLogger(f"gplus_trees.{name}")


def get_test_logger(name: str) -> logging.Logger:
    """
    Get a logger for tests with appropriate configuration.
    
    Args:
        name: Test module name
        
    Returns:
        Logger instance for tests
    """
    logger = logging.getLogger(f"Tests.{name}")
    
    # Set up test-specific logging if not already done
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    
    return logger
