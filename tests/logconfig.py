# logconfig.py
"""Test logging configuration - now using centralized logging."""

import sys
import os

# Add src to path to import logging config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from gplus_trees.logging_config import get_test_logger

# Get a logger for tests
logger = get_test_logger("main")
