"""Centralized logging configuration for the gplus-trees project.

Best-practice notes
-------------------
* **Library code** (everything under ``src/gplus_trees/``) must never
  attach handlers, set levels, or call ``logging.basicConfig()``.
  It should only obtain a logger via ``logging.getLogger(__name__)``
  and emit records.  The *application* entry-point (a script, a test
  runner, ``pytest``, …) is responsible for configuring handlers,
  levels, and formatters.

* A single ``NullHandler`` is attached to the package-level logger
  ``"gplus_trees"`` so that library consumers who do **not** configure
  logging never see the "No handlers could be found" warning.

* The convenience function :func:`get_logger` is provided so that
  every module in the package can simply write::

      from gplus_trees.logging_config import get_logger
      logger = get_logger(__name__)

  …and will get a properly namespaced logger without any side-effects.
"""

import logging

# Attach a NullHandler to the library root logger.  This is the
# recommended practice for library packages (see Python docs,
# "Configuring Logging for a Library").  It ensures that log records
# produced by the library are silently discarded unless the
# *application* has configured logging.
logging.getLogger("gplus_trees").addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module *name*.

    Typical usage at the top of every module::

        from gplus_trees.logging_config import get_logger
        logger = get_logger(__name__)

    If *name* already starts with ``"gplus_trees."`` (which is the
    case when ``__name__`` is used inside the package), the name is
    passed through unchanged.  Otherwise it is prefixed with
    ``"gplus_trees."`` so that the resulting logger is always a child
    of the package-level logger.

    Parameters
    ----------
    name : str
        Module name - almost always ``__name__``.

    Returns
    -------
    logging.Logger
    """
    if name.startswith("gplus_trees"):
        return logging.getLogger(name)
    return logging.getLogger(f"gplus_trees.{name}")
