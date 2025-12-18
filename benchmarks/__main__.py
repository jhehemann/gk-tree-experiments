"""Allow running benchmarks as: python -m benchmarks"""

from .run_benchmarks import main
import sys

if __name__ == "__main__":
    sys.exit(main())
