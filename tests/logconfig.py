# logconfig.py
"""Test logging configuration - now using centralized logging."""

from gplus_trees.logging_config import get_test_logger

# Get a logger for tests
logger = get_test_logger("main")
