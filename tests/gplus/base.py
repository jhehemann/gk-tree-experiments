"""Tests for G+-trees with factory pattern"""

# Import the unified base class
from tests.test_base import GPlusTreeTestCase

# Inherit from the unified base class instead of duplicating code
class TreeTestCase(GPlusTreeTestCase):
    """Legacy TreeTestCase that inherits from unified GPlusTreeTestCase."""
    pass
