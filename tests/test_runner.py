#!/usr/bin/env python3
# filepath: /Users/jannikhehemann/coding/data_structures/gplus-trees/tests/test_selection.py
"""
Simple test runner that collects and runs specified test files from the project.
"""

import argparse
import logging
import os
import sys
import unittest

# Test files to run - Add or remove files as needed
TEST_FILES = [
    "gplus/test_gplus_tree_base.py",
    "gplus/test_insert.py",
    "gplus/test_retrieve.py",
    "merkle/test_gp_mkl_tree.py",
    "gk_plus/test_counts_and_sizes.py",
    "gk_plus/test_expanded_count.py",
    "gk_plus/test_get_max_dim.py",
    "gk_plus/test_insert_dimensions.py",
    "gk_plus/test_insert.py",
    "gk_plus/test_item_count.py",
    "gk_plus/test_set_conversion.py",
    "gk_plus/test_gk_plus_zip.py",
    "gk_plus/test_gk_plus_unzip.py",
    "test_klist.py",
    "test_insert_result.py",
    "test_rank_hash_consistency.py",
    "test_stats_gpltree.py",
    "test_utility_functions.py",
]


def run_tests(test_files=None, test_classes=None, verbosity=0):
    """
    Run specified test files and optionally specific test classes.

    Args:
        test_files: List of test files to run (can include :: syntax for specific tests)
        test_classes: Optional dict mapping test files to specific test classes to run
        verbosity: Verbosity level for test output

    Returns:
        Test result object
    """
    if test_files is None:
        test_files = TEST_FILES

    if test_classes is None:
        test_classes = {}

    suite = unittest.TestSuite()

    # Base test directory
    os.path.dirname(__file__)

    for test_spec in test_files:
        # Parse the test specification
        if "::" in test_spec:
            # Handle specific test method syntax: file.py::TestClass::test_method
            parts = test_spec.split("::")
            file_path = parts[0]
            test_class = parts[1] if len(parts) > 1 else None
            test_method = parts[2] if len(parts) > 2 else None
        else:
            # Regular file path
            file_path = test_spec
            test_class = None
            test_method = None

        # Build module name from file path
        if "/" in file_path:
            path_parts = file_path.split("/")
            module_name = f"tests.{'.'.join(path_parts)}".replace(".py", "")
        else:
            module_name = f"tests.{file_path}".replace(".py", "")

        # Load the appropriate tests
        if test_method and test_class:
            # Load specific test method
            test_name = f"{module_name}.{test_class}.{test_method}"
            suite.addTest(unittest.TestLoader().loadTestsFromName(test_name))
        elif test_class:
            # Load specific test class
            test_name = f"{module_name}.{test_class}"
            suite.addTest(unittest.TestLoader().loadTestsFromName(test_name))
        elif file_path in test_classes:
            # Load specific classes specified in test_classes parameter
            for class_name in test_classes[file_path]:
                suite.addTest(unittest.TestLoader().loadTestsFromName(f"{module_name}.{class_name}"))
        else:
            # Load all tests from the file
            suite.addTest(unittest.TestLoader().loadTestsFromName(module_name))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result


def main():
    """Main entry point for running tests."""
    parser = argparse.ArgumentParser(description="Run specific tests for G+ Trees project")

    parser.add_argument(
        "-f", "--files", nargs="+", help="Test files to run (e.g., test_klist.py gk_plus/test_split_inplace.py)"
    )

    parser.add_argument(
        "-c", "--classes", nargs="+", help="Test classes to run (format: file.py:TestClass1,TestClass2)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2, 3], default=1, help="Verbosity level (0-3)")

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level)
    # Force=True makes this configuration override existing logger settings
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True)

    # Use command line arguments if provided, otherwise use defaults from TEST_FILES
    test_files = args.files if args.files else TEST_FILES

    # Parse test classes if provided
    test_classes = {}
    if args.classes:
        for class_arg in args.classes:
            if ":" in class_arg:
                file_name, class_names = class_arg.split(":")
                test_classes[file_name] = class_names.split(",")

    result = run_tests(test_files=test_files, test_classes=test_classes, verbosity=args.verbosity)

    # Exit with non-zero code if tests failed
    sys.exit(1 if (result.failures or result.errors) else 0)


if __name__ == "__main__":
    main()
