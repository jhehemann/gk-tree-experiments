"""Test logging configuration.

This module exists only for backward compatibility.  New test modules
should obtain their logger directly via::

    import logging
    logger = logging.getLogger(__name__)

Logging *configuration* (level, format, handlers) is handled by pytest
through ``[tool.pytest.ini_options]`` in ``pyproject.toml``.
"""

import logging

logger = logging.getLogger(__name__)
