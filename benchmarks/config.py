"""Benchmark configuration and metadata management."""

import os
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    # Reproducibility
    seed: int = 42
    
    # Benchmark parameters
    sizes: list[int] = None
    ks: list[int] = None
    repetitions: int = 200
    
    # Execution control
    verify_only: bool = False
    skip_warmup: bool = False
    
    # Logging
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.sizes is None:
            self.sizes = [10, 100, 1000]
        if self.ks is None:
            self.ks = [2, 16]
    
    @classmethod
    def from_env(cls) -> "BenchmarkConfig":
        """Create config from environment variables."""
        return cls(
            seed=int(os.environ.get("BENCHMARK_SEED", "42")),
            verify_only=os.environ.get("BENCHMARK_VERIFY_ONLY", "").lower() == "true",
            skip_warmup=os.environ.get("BENCHMARK_SKIP_WARMUP", "").lower() == "true",
            log_level=os.environ.get("BENCHMARK_LOG_LEVEL", "INFO"),
        )


def get_git_commit_hash() -> Optional[str]:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@dataclass
class BenchmarkMetadata:
    """Metadata about a benchmark run."""
    
    commit_hash: Optional[str]
    config: BenchmarkConfig
    size: int
    k: int
    repetitions: int
    
    def __str__(self) -> str:
        """Format metadata as string."""
        lines = [
            f"Commit: {self.commit_hash or 'unknown'}",
            f"Seed: {self.config.seed}",
            f"Size (n): {self.size}",
            f"Target node size (K): {self.k}",
            f"Repetitions: {self.repetitions}",
        ]
        return "\n".join(lines)
