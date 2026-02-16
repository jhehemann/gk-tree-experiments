#!/usr/bin/env python3
"""
Isolated Background Benchmark Runner

Runs ASV benchmarks in a completely isolated directory clone,
in the background, without interfering with the working directory.

Architecture
------------
- RepoManager               - git clone / fetch / branch-sync for the isolated repo
- ASVEnvironment             - ASV config generation, dependency bootstrap, config patching
- StatusManager              - status.json bookkeeping, log-file creation, progress display
- IsolatedBenchmarkRunner    - top-level orchestrator that composes the three above

Security note
-------------
Benchmarking a commit means *executing* Python code from that commit
(via ``pip install`` and ASV's benchmark harness).  Never benchmark
untrusted branches on a machine you care about.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _run_command(
    cmd: list[str],
    cwd: str | Path | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run *cmd*, print diagnostics on failure, and return the result."""
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=capture_output, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd)}")
        print(f"Error: {e.stderr if e.stderr else e}")
        raise


# ---------------------------------------------------------------------------
# RepoManager
# ---------------------------------------------------------------------------


class RepoManager:
    """Git operations for the isolated benchmark repository."""

    def __init__(self, working_dir: Path, repo_dir: Path) -> None:
        self.working_dir = working_dir
        self.repo_dir = repo_dir

    # -- low-level helpers --------------------------------------------------

    def remote_branch_exists(self, branch_name: str) -> bool:
        """Return ``True`` when ``origin/<branch_name>`` exists."""
        return (
            subprocess.run(
                ["git", "show-ref", "--verify", f"refs/remotes/origin/{branch_name}"],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                check=False,
            ).returncode
            == 0
        )

    def sync_branch_with_remote(self, branch_name: str) -> None:
        """Checkout *branch_name* and hard-reset to origin when available."""
        try:
            _run_command(["git", "checkout", branch_name], cwd=self.repo_dir)
        except subprocess.CalledProcessError:
            if self.remote_branch_exists(branch_name):
                _run_command(
                    ["git", "checkout", "-b", branch_name, f"origin/{branch_name}"],
                    cwd=self.repo_dir,
                )
            else:
                _run_command(
                    ["git", "checkout", "-b", branch_name],
                    cwd=self.repo_dir,
                )
                print(f"⚠️  Created local branch '{branch_name}' (no origin/{branch_name} exists)")

        if self.remote_branch_exists(branch_name):
            _run_command(
                ["git", "reset", "--hard", f"origin/{branch_name}"],
                cwd=self.repo_dir,
            )
        else:
            print(f"⚠️  Remote branch origin/{branch_name} not found; using local '{branch_name}' head")

    def current_working_branch(self) -> str:
        """Return the branch name of the *working* directory."""
        return _run_command(["git", "branch", "--show-current"], cwd=self.working_dir).stdout.strip()

    # -- high-level operations ----------------------------------------------

    def clone_or_fetch(self, current_branch: str) -> None:
        """Clone the repo on first run, or fetch + sync on subsequent runs."""
        if self.repo_dir.exists():
            print("📡 Updating existing isolated repository...")
            _run_command(["git", "fetch", "--all"], cwd=self.repo_dir)
            self.sync_branch_with_remote(current_branch)
            self._print_head("Isolated repo updated to commit")
        else:
            print("📥 Cloning repository for isolated benchmarking...")
            url = _run_command(["git", "remote", "get-url", "origin"], cwd=self.working_dir).stdout.strip()
            _run_command(
                ["git", "clone", url, str(self.repo_dir)],
                cwd=self.repo_dir.parent,
            )
            _run_command(["git", "fetch", "--all"], cwd=self.repo_dir)
            self.sync_branch_with_remote(current_branch)
            self._print_head("Isolated repo cloned at commit")

    def setup_benchmark_branches(self, current_branch: str) -> list[str]:
        """Create local tracking branches for every remote; return the list."""
        print("🔄 Setting up local benchmark branches...")
        remote_branches: list[str] = []
        for line in _run_command(["git", "branch", "-r"], cwd=self.repo_dir).stdout.strip().split("\n"):
            branch = line.strip()
            if "->" not in branch and branch.startswith("origin/"):
                name = branch.removeprefix("origin/")
                if name not in ("HEAD", "main", "develop"):
                    remote_branches.append(name)

        benchmark_branches = list(set([current_branch, *remote_branches]))
        print("🔍 Found branches for benchmarking: " + ", ".join(sorted(benchmark_branches)))

        for branch in benchmark_branches:
            try:
                _run_command(
                    ["git", "show-ref", "--verify", f"refs/heads/{branch}"],
                    cwd=self.repo_dir,
                )
            except subprocess.CalledProcessError:
                try:
                    _run_command(
                        ["git", "checkout", "-b", branch, f"origin/{branch}"],
                        cwd=self.repo_dir,
                    )
                    print(f"✅ Created local branch '{branch}' from origin/{branch}")
                except subprocess.CalledProcessError:
                    print(f"⚠️  Could not create local branch '{branch}' - remote may not exist")

        self.sync_branch_with_remote(current_branch)
        return benchmark_branches

    def force_update(self) -> bool:
        """Fetch + hard-reset to match origin.  Return success."""
        if not self.repo_dir.exists():
            print("❌ Isolated repository not found. Run --setup first.")
            return False

        print("🔄 Force updating isolated repository to latest remote state...")
        try:
            branch = self.current_working_branch()
        except subprocess.CalledProcessError:
            branch = "main"
        print(f"📍 Working directory branch: {branch}")

        try:
            _run_command(["git", "fetch", "--all"], cwd=self.repo_dir)
            self.sync_branch_with_remote(branch)
            self._print_head(f"Isolated repo updated to latest {branch}")

            try:
                work_sha = _run_command(["git", "rev-parse", "HEAD"], cwd=self.working_dir).stdout.strip()
                iso_sha = _run_command(["git", "rev-parse", "HEAD"], cwd=self.repo_dir).stdout.strip()
                if work_sha == iso_sha:
                    print("✅ Isolated repo and working directory are in sync")
                else:
                    print("💡 Different commits - normal if you have uncommitted changes")
            except subprocess.CalledProcessError:
                pass

            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to update repository: {e}")
            return False

    # -- internal -----------------------------------------------------------

    def _print_head(self, label: str) -> None:
        try:
            sha = _run_command(["git", "rev-parse", "HEAD"], cwd=self.repo_dir).stdout.strip()
            print(f"📍 {label}: {sha[:12]}...")
        except subprocess.CalledProcessError:
            print("⚠️  Could not verify current commit")


# ---------------------------------------------------------------------------
# ASVEnvironment
# ---------------------------------------------------------------------------


class ASVEnvironment:
    """ASV configuration, dependency bootstrap, and config patching."""

    INSTALL_CMD = "in-dir={build_dir} python -m pip install . --force-reinstall"

    def __init__(
        self,
        repo_dir: Path,
        benchmark_dir: Path,
        results_dir: Path,
        repo_manager: RepoManager,
    ) -> None:
        self.repo_dir = repo_dir
        self.benchmark_dir = benchmark_dir
        self.results_dir = results_dir
        self._repo = repo_manager

    @property
    def config_path(self) -> Path:
        return self.repo_dir / "asv.conf.json"

    # -- bootstrap (setup-time only) ----------------------------------------

    def bootstrap(self) -> None:
        """Install Poetry dev deps and verify ASV.  Called during setup."""
        print("🔧 Bootstrapping benchmark tooling (asv)...")
        try:
            _run_command(["poetry", "install", "--with", "dev"], cwd=self.repo_dir)
        except subprocess.CalledProcessError:
            print("🔒 Poetry lock file is out of date; regenerating...")
            _run_command(["poetry", "lock"], cwd=self.repo_dir)
            _run_command(["poetry", "install", "--with", "dev"], cwd=self.repo_dir)
        _run_command(["poetry", "run", "asv", "--version"], cwd=self.repo_dir)
        print("✅ ASV installed successfully")

    def verify_available(self) -> None:
        """Fast check that ASV is importable.  Fails fast on the run path."""
        try:
            _run_command(["poetry", "run", "asv", "--version"], cwd=self.repo_dir)
        except subprocess.CalledProcessError:
            raise RuntimeError(
                "ASV is not available in the isolated environment. Run './benchmark setup' first."
            ) from None

    # -- config management ---------------------------------------------------

    def write_config(self, branches: list[str]) -> None:
        """Write a fresh ``asv.conf.json`` for the isolated environment."""
        cfg = {
            "version": 1,
            "project": "gk-tree-experiments-isolated",
            "project_url": "isolated-benchmarking",
            "repo": str(self.repo_dir),
            "environment_type": "virtualenv",
            "benchmark_dir": "benchmarks",
            "env_dir": str(self.benchmark_dir / ".asv" / "env"),
            "results_dir": str(self.results_dir),
            "html_dir": str(self.benchmark_dir / "html"),
            "pythons": ["3.11"],
            "matrix": {"numpy": [""]},
            "build_command": [],
            "install_command": [self.INSTALL_CMD],
            "dvcs": "git",
            "branches": branches,
        }
        with open(self.config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    def ensure_install_config(self) -> None:
        """Patch existing ``asv.conf.json`` build/install commands."""
        if not self.config_path.exists():
            return
        try:
            with open(self.config_path) as f:
                cfg = json.load(f)
            changed = False
            if cfg.get("build_command") != []:
                cfg["build_command"] = []
                changed = True
            if cfg.get("install_command") != [self.INSTALL_CMD]:
                cfg["install_command"] = [self.INSTALL_CMD]
                changed = True
            if changed:
                with open(self.config_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                print("✅ Updated ASV build/install commands for Poetry compatibility")
        except Exception as e:
            print(f"⚠️  Could not ensure ASV install configuration: {e}")

    def add_branch(self, branch_name: str) -> None:
        """Ensure *branch_name* appears in ``asv.conf.json`` branches."""
        if not self.config_path.exists():
            return
        try:
            with open(self.config_path) as f:
                cfg = json.load(f)
            branches = cfg.get("branches", [])
            if branch_name not in branches:
                branches.append(branch_name)
                cfg["branches"] = branches
                with open(self.config_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                print(f"✅ Added '{branch_name}' to ASV branches configuration")
        except Exception as e:
            print(f"⚠️  Could not update ASV config: {e}")

    def sync_branches_with_results(self) -> None:
        """Discover branches that have results and update ASV config."""
        print("🔍 Ensuring all branches are included in HTML generation...")

        # Collect result files across all machine directories
        results_files: list[Path] = []
        for d in self.results_dir.iterdir():
            if d.is_dir():
                results_files.extend(d.glob("*.json"))
        if not results_files:
            print("⚠️  No machine-specific results found")
            return

        branches_with_results: set[str] = set()
        commit_hashes: set[str] = set()
        for rf in results_files:
            if rf.name == "machine.json":
                continue
            try:
                data = json.loads(rf.read_text())
                sha = data.get("commit_hash")
                if not sha:
                    continue
                commit_hashes.add(sha)
                try:
                    br = _run_command(
                        ["git", "branch", "--contains", sha],
                        cwd=self.repo_dir,
                    )
                    for line in br.stdout.strip().split("\n"):
                        name = line.strip().lstrip("* ")
                        if name and not name.startswith("("):
                            branches_with_results.add(name)
                except subprocess.CalledProcessError:
                    pass
            except (json.JSONDecodeError, OSError):
                pass

        print(
            f"📊 Found results for {len(commit_hashes)} commits across "
            f"{len(branches_with_results)} branches: "
            f"{', '.join(sorted(branches_with_results))}"
        )

        if not branches_with_results:
            return

        self._reconcile_branches(branches_with_results)

        print("🔄 Rebuilding ASV data to include all branches...")
        try:
            _run_command(["poetry", "run", "asv", "publish"], cwd=self.repo_dir)
            print("✅ HTML successfully regenerated with all branches")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to publish HTML: {e}")

    # -- internal -----------------------------------------------------------

    def _reconcile_branches(self, branches_with_results: set[str]) -> None:
        """Sync ``asv.conf.json`` branches with what actually exists."""
        if not self.config_path.exists():
            return
        try:
            with open(self.config_path) as f:
                cfg = json.load(f)
            current = set(cfg.get("branches", []))

            _run_command(["git", "fetch", "origin", "--prune"], cwd=self.repo_dir)

            existing_remote: set[str] = set()
            stale: list[str] = []
            for b in current | branches_with_results:
                if self._repo.remote_branch_exists(b):
                    existing_remote.add(b)
                elif b in current:
                    stale.append(b)

            for b in stale:
                print(f"🗑️  Branch '{b}' no longer exists in origin, removing")
                try:
                    _run_command(
                        ["git", "show-ref", "--verify", f"refs/heads/{b}"],
                        cwd=self.repo_dir,
                    )
                    _run_command(["git", "branch", "-d", b], cwd=self.repo_dir)
                except subprocess.CalledProcessError:
                    pass

            desired = existing_remote & (current | branches_with_results)
            if desired != current:
                cfg["branches"] = sorted(desired)
                with open(self.config_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                added = desired - current
                removed = current - desired
                if added:
                    print(f"✅ Added branches: {', '.join(sorted(added))}")
                if removed:
                    print(f"🗑️  Removed branches: {', '.join(sorted(removed))}")
        except Exception as e:
            print(f"⚠️  Could not reconcile ASV branches: {e}")


# ---------------------------------------------------------------------------
# StatusManager
# ---------------------------------------------------------------------------


class StatusManager:
    """Benchmark status tracking, log-file creation, and progress display."""

    def __init__(self, status_file: Path, logs_dir: Path) -> None:
        self.status_file = status_file
        self.logs_dir = logs_dir

    # -- status CRUD --------------------------------------------------------

    def update(
        self,
        status: str,
        message: str = "",
        commit: str | None = None,
        pid: int | None = None,
        pgid: int | None = None,
        run_id: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        data = {
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "commit": commit,
            "pid": pid or os.getpid(),
            "pgid": pgid,
            "run_id": run_id,
            "exit_code": exit_code,
        }
        # Atomic write: temp file + os.replace() prevents corruption on crash
        tmp = self.status_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.status_file)

    def get(self) -> dict:
        if not self.status_file.exists():
            return {"status": "idle", "message": "No benchmarks running"}
        try:
            with open(self.status_file) as f:
                st = json.load(f)
            if st.get("status") == "running" and "pid" in st:
                try:
                    os.kill(st["pid"], 0)
                except OSError:
                    # Process died; check if it wrote an exit code
                    exit_code = st.get("exit_code")
                    if exit_code is not None:
                        if exit_code == 0:
                            st["status"] = "completed"
                            st["message"] = "Benchmarks completed successfully for " + st.get("commit", "unknown")
                        else:
                            st["status"] = "failed"
                            st["message"] = (
                                f"Benchmarks failed with exit code {exit_code} for {st.get('commit', 'unknown')}"
                            )
                    else:
                        # No exit code (e.g., killed externally) — fall back to log check
                        if self._check_completion():
                            st["status"] = "completed"
                            st["message"] = "Benchmarks completed successfully for " + st.get("commit", "unknown")
                        else:
                            st["status"] = "failed"
                            st["message"] = "Process terminated unexpectedly"
                    # Atomic write of updated status
                    tmp = self.status_file.with_suffix(".tmp")
                    with open(tmp, "w") as f:
                        json.dump(st, f, indent=2)
                    os.replace(tmp, self.status_file)
            return st
        except Exception:
            return {"status": "unknown", "message": "Could not read status"}

    def show(self) -> None:
        st = self.get()
        print(f"📊 Benchmark Status: {st['status'].upper()}")
        print(f"💬 Message: {st.get('message', 'No message')}")
        print(f"⏰ Last Update: {st.get('timestamp', 'Unknown')}")
        if st.get("commit"):
            print(f"🔍 Commit: {st['commit']}")
        if st.get("run_id"):
            print(f"🆔 Run ID: {st['run_id']}")

        logs = self._recent_logs(3)
        if logs:
            print("\n📋 Recent Logs:")
            for lf in logs:
                print(f"  - {lf.name}")

        if st.get("status") == "running" and logs:
            print(f"\n🖥️  Recent output from {logs[0].name}:")
            try:
                tail = logs[0].read_text().splitlines()[-5:]
                for line in tail:
                    print(f"  {line.rstrip()}")
            except Exception:
                pass

    # -- log files -----------------------------------------------------------

    def create_log_file(self, commit_hash: str, run_id: str) -> Path:
        """Return a fresh log path incorporating the *run_id*."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = commit_hash.replace("..", "_to_").replace("^!", "_head").replace("/", "_").replace(" ", "_")
        return self.logs_dir / f"benchmark_{safe}_{run_id[:8]}_{ts}.log"

    # -- internal -----------------------------------------------------------

    def _recent_logs(self, n: int) -> list[Path]:
        return sorted(
            self.logs_dir.glob("*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:n]

    def _check_completion(self) -> bool:
        logs = self._recent_logs(1)
        if not logs:
            return False
        try:
            lines = [ln.rstrip() for ln in logs[0].read_text().splitlines() if ln.strip()]
            if not lines:
                return False
            last = lines[-1]
            return all(c == "=" or c.isspace() for c in last) and "=" in last
        except Exception:
            return False


# ---------------------------------------------------------------------------
# _TeeWriter - duplicate writes to two streams
# ---------------------------------------------------------------------------


class _TeeWriter:
    """Write to two file-like objects simultaneously."""

    def __init__(self, stream, logfile):
        self._stream = stream
        self._logfile = logfile

    def write(self, data):
        if self._stream:
            self._stream.write(data)
            self._stream.flush()
        self._logfile.write(data)
        self._logfile.flush()

    def flush(self):
        if self._stream:
            self._stream.flush()
        self._logfile.flush()

    def fileno(self):
        return self._logfile.fileno()


# ---------------------------------------------------------------------------
# IsolatedBenchmarkRunner (orchestrator)
# ---------------------------------------------------------------------------


class IsolatedBenchmarkRunner:
    """Top-level orchestrator composing RepoManager, ASVEnvironment,
    and StatusManager."""

    def __init__(self, working_dir: str | None = None) -> None:
        self.working_dir = Path(working_dir or os.getcwd())
        self.benchmark_dir = self.working_dir.parent / ".isolated-benchmarks"
        self.repo_dir = self.benchmark_dir / "gk-tree-experiments"
        self.results_dir = self.benchmark_dir / "results"
        self.logs_dir = self.benchmark_dir / "logs"

        self.benchmark_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        self.repo = RepoManager(self.working_dir, self.repo_dir)
        self.asv = ASVEnvironment(self.repo_dir, self.benchmark_dir, self.results_dir, self.repo)
        self.status = StatusManager(self.benchmark_dir / "status.json", self.logs_dir)

    # -- setup ---------------------------------------------------------------

    def setup_isolated_repo(self) -> None:
        """Full first-time (or refresh) setup of the isolated environment."""
        print("🔧 Setting up isolated benchmark environment...")
        branch = self.repo.current_working_branch()
        print(f"📍 Current branch: {branch}")

        self.repo.clone_or_fetch(branch)
        branches = self.repo.setup_benchmark_branches(branch)

        self.asv.write_config(branches)
        self.asv.ensure_install_config()
        self.asv.bootstrap()

        print("✅ Isolated environment ready!")

    def remove_git_hooks(self) -> None:
        hook = self.working_dir / ".git" / "hooks" / "post-commit"
        if hook.exists():
            hook.unlink()
            print("🗑️  Removed existing post-commit git hook")
        else:
            print("i  No post-commit git hook to remove")

    # -- run (worker) --------------------------------------------------------

    def run_benchmarks_worker(
        self,
        commit_hash: str,
        benchmark_filter: str | None = None,
        branch: str | None = None,
        quick: bool = False,
    ) -> None:
        """Foreground worker - called inside the detached subprocess.

        All output (including bootstrap diagnostics) is tee'd to the
        run-specific log file from the very start so nothing is lost.
        """
        run_id = uuid.uuid4().hex
        log_file = self.status.create_log_file(commit_hash, run_id)

        with open(log_file, "w") as tee:
            tee.write(f"Benchmark run started at {datetime.now()}\n")
            tee.write(f"Run ID: {run_id}\n")
            tee.write(f"Original commit reference: {commit_hash}\n")
            tee.write(f"Filter: {benchmark_filter or 'All benchmarks'}\n")
            tee.write(f"Quick mode: {quick}\n")
            tee.write("=" * 50 + "\n\n")
            tee.flush()

            # Mirror stdout/stderr to the log file so bootstrap output is captured
            sys.stdout = _TeeWriter(sys.__stdout__, tee)
            sys.stderr = _TeeWriter(sys.__stderr__, tee)

            try:
                resolved_commit, resolved_branch = self._resolve_commit_reference(commit_hash, branch)
                effective_branch = branch or resolved_branch

                self.status.update(
                    "running",
                    f"Starting benchmarks for {commit_hash}",
                    commit_hash,
                    run_id=run_id,
                )

                # ---- fetch + sync ----
                print("📡 Updating isolated repository to latest remote state...")
                _run_command(["git", "fetch", "--all"], cwd=self.repo_dir)

                target_branch = effective_branch
                if not target_branch:
                    try:
                        target_branch = self.repo.current_working_branch()
                        print(f"📍 Using working directory branch: {target_branch}")
                    except subprocess.CalledProcessError:
                        target_branch = "main"
                        print(f"📍 Falling back to default branch: {target_branch}")

                print(f"🔄 Updating to latest {target_branch}...")
                try:
                    self.repo.sync_branch_with_remote(target_branch)
                    self.asv.add_branch(target_branch)
                    sha = _run_command(["git", "rev-parse", "HEAD"], cwd=self.repo_dir).stdout.strip()
                    print(f"✅ Isolated repo at {target_branch}: {sha[:12]}...")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to update branch {target_branch}: {e}")
                    print("💡 Benchmarks may run on stale code")

                # ---- ASV environment ----
                self.status.update(
                    "running",
                    "Setting up ASV environment...",
                    commit_hash,
                    run_id=run_id,
                )
                self.asv.ensure_install_config()
                self.asv.verify_available()
                _run_command(
                    ["poetry", "run", "asv", "machine", "--yes"],
                    cwd=self.repo_dir,
                )

                # ---- build ASV command ----
                bench_cmd = ["poetry", "run", "asv", "run"]
                if quick:
                    bench_cmd.append("--quick")
                bench_cmd.extend(["--python=3.11", resolved_commit])
                if benchmark_filter:
                    bench_cmd.extend(["--bench", benchmark_filter])

                self.status.update(
                    "running",
                    f"Running benchmarks for {commit_hash}...",
                    commit_hash,
                    run_id=run_id,
                )

                tee.write(f"\nResolved commit reference: {resolved_commit}\n")
                tee.write(f"ASV command: {' '.join(bench_cmd)}\n")
                tee.write("=" * 50 + "\n\n")
                tee.flush()

                # ---- execute ASV ----
                process = subprocess.Popen(
                    bench_cmd,
                    cwd=self.repo_dir,
                    stdout=tee,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self.status.update(
                    "running",
                    f"Running benchmarks for {commit_hash}...",
                    commit_hash,
                    pid=process.pid,
                    run_id=run_id,
                )
                rc = process.wait()

                # ---- post-processing ----
                if rc == 0:
                    self.status.update(
                        "running",
                        "Generating HTML report...",
                        commit_hash,
                        run_id=run_id,
                        exit_code=0,
                    )
                    _run_command(["poetry", "run", "asv", "publish"], cwd=self.repo_dir)
                    self.status.update(
                        "completed",
                        f"Benchmarks completed successfully for {commit_hash}",
                        commit_hash,
                        run_id=run_id,
                        exit_code=0,
                    )
                    print(f"✅ Benchmarks completed for {commit_hash}")
                    print(f"📊 Results: {self.benchmark_dir / 'html' / 'index.html'}")
                    print(f"📋 Log: {log_file}")
                else:
                    self.status.update(
                        "failed",
                        f"Benchmarks failed for {commit_hash}",
                        commit_hash,
                        run_id=run_id,
                        exit_code=rc,
                    )
                    print(f"❌ Benchmarks failed for {commit_hash}")
                    print(f"📋 Check log: {log_file}")

            except Exception as e:
                msg = f"Benchmark error: {e}"
                self.status.update("failed", msg, commit_hash, run_id=run_id, exit_code=1)
                print(f"❌ {msg}")
            finally:
                tee.flush()
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__

    # -- run (background launcher) -------------------------------------------

    def run_benchmarks_background(
        self,
        commit_hash: str,
        benchmark_filter: str | None = None,
        branch: str | None = None,
        quick: bool = False,
    ) -> subprocess.Popen:
        """Launch a detached worker subprocess.

        The child runs in its own *process group* (``start_new_session=True``)
        so ``stop_benchmarks`` can cleanly terminate the entire tree via
        ``os.killpg`` instead of scanning ``ps aux``.
        """
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--run",
            commit_hash,
            "--working-dir",
            str(self.working_dir),
        ]
        if benchmark_filter:
            cmd.extend(["--bench", benchmark_filter])
        if branch:
            cmd.extend(["--branch", branch])
        if quick:
            cmd.append("--quick")

        proc = subprocess.Popen(
            cmd,
            cwd=self.working_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # start_new_session=True ⟹ proc.pid == pgid
        self.status.update(
            "running",
            f"Queued benchmarks for {commit_hash}",
            commit_hash,
            pid=proc.pid,
            pgid=proc.pid,
        )

        if ".." in commit_hash:
            kind = "commit range"
        elif " " in commit_hash.strip():
            kind = "multiple commits"
        else:
            kind = "commit"
        print(f"🚀 Benchmarks started in background for {kind}: {commit_hash}")
        print(f"📊 Monitor progress: python {__file__} --status")
        print(f"🔍 View results when done: python {__file__} --view")
        return proc

    # -- stop (process-group based) ------------------------------------------

    def stop_benchmarks(self) -> None:
        """Terminate the benchmark process group tracked in ``status.json``.

        Uses ``os.killpg`` when a pgid is available (the normal case),
        falling back to single-PID termination for backward compatibility
        with older status files that lack a pgid.
        """
        st = self.status.get()
        if st["status"] != "running":
            print(f"i  No benchmarks currently running (status: {st['status']})")
            return

        pgid = st.get("pgid")
        pid = st.get("pid")

        if pgid:
            self._kill_process_group(pgid)
        elif pid:
            self._kill_single_process(pid)
        else:
            print("❌ No process ID found in status")
            return

        self.status.update("stopped", "Benchmark stopped by user", st.get("commit"))
        print("✅ Benchmark process stopped successfully")

    # -- view ----------------------------------------------------------------

    def view_results(self) -> None:
        """Open benchmark results in browser using a local web server."""
        if not (self.results_dir / "benchmarks.json").exists():
            print("❌ No results found. Run benchmarks first.")
            return

        html_dir = self.benchmark_dir / "html"
        print("📊 Ensuring HTML includes all available branch results...")
        try:
            self.asv.ensure_install_config()
            self.asv.verify_available()
            self.asv.sync_branches_with_results()
            _run_command(["poetry", "run", "asv", "publish"], cwd=self.repo_dir)
            print("✅ HTML results generated successfully")
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"❌ Failed to generate HTML results: {e}")
            print("💡 Try running: ./benchmark setup")
            return

        print("🌐 Starting local web server at port 8081...")
        print("📝 Press Ctrl+C to stop the web server")
        try:
            _run_command(
                [
                    "poetry",
                    "run",
                    "asv",
                    "preview",
                    "--browser",
                    "--port",
                    "8081",
                    "--html-dir",
                    str(html_dir),
                ],
                cwd=self.repo_dir,
            )
        except subprocess.CalledProcessError:
            html_file = html_dir / "index.html"
            if html_file.exists():
                import webbrowser

                webbrowser.open(f"file://{html_file}")
                print(f"📂 Opened: {html_file}")
        except KeyboardInterrupt:
            print("\n🛑 Web server stopped")

    # -- clean ---------------------------------------------------------------

    def clean_benchmarks(self, force: bool = False) -> bool:
        if not force:
            print("⚠️  This will permanently delete:")
            print("   • All benchmark results and data, including all potentially generated adversarial keys.")
            print("   • Generated HTML reports")
            print("   • Benchmark execution logs")
            print("   • Isolated repository clone")
            print("   • Git hooks")
            print("After cleanup, you will need to re-run setup to create a new environment.\n")
            resp = input("Are you sure you want to continue? (y/N): ")
            if resp.strip().lower() not in ("y", "yes"):
                print("❌ Operation cancelled")
                return False

        print("🧹 Cleaning up isolated benchmark environment...")
        if self.benchmark_dir.exists():
            shutil.rmtree(self.benchmark_dir)
            print(f"✅ Cleaned up {self.benchmark_dir}")
        else:
            print("i  No benchmark directory to clean")

        # Remove git hooks
        hook = self.working_dir / ".git" / "hooks" / "post-commit"
        if hook.exists():
            hook.unlink()
            print("✅ Removed git hooks")
        else:
            print("i  No git hooks to remove")
        return True

    # -- force update --------------------------------------------------------

    def force_update_repo(self) -> bool:
        return self.repo.force_update()

    # -- commit resolution (private) ----------------------------------------

    def _resolve_commit_reference(self, commit_ref: str, branch: str | None = None) -> tuple[str, str | None]:
        if " " in commit_ref.strip():
            refs: list[str] = []
            branches: list[str] = []
            for part in commit_ref.strip().split():
                r, b = self._resolve_single(part.strip(), branch)
                refs.append(r)
                if b:
                    branches.append(b)
            final = " ".join(refs)
            print(f"🔍 Resolved multiple commits: {final}")
            return final, branches[0] if branches else None
        return self._resolve_single(commit_ref, branch)

    def _resolve_single(self, commit_ref: str, branch: str | None) -> tuple[str, str | None]:
        if commit_ref.endswith("^!"):
            base = commit_ref[:-2]
            if base == "HEAD":
                print(f"🔍 Using {commit_ref} as-is for ASV")
                return commit_ref, None
            for ref in (base, f"origin/{base}"):
                try:
                    sha = _run_command(["git", "rev-parse", ref], cwd=self.working_dir).stdout.strip()
                    resolved = f"{sha}^!"
                    print(f"🔍 Resolved {commit_ref} → {resolved[:20]}...")
                    return (
                        resolved,
                        base if ref.startswith("origin/") else None,
                    )
                except subprocess.CalledProcessError:
                    continue
            print(f"⚠️  Could not resolve {commit_ref}, using base: {base}")
            return base, None

        print(f"🔍 Passing {commit_ref} directly to ASV")
        return commit_ref, None

    # -- internal process management -----------------------------------------

    @staticmethod
    def _kill_process_group(pgid: int) -> None:
        """Terminate an entire process group (SIGTERM → SIGKILL fallback)."""
        try:
            os.killpg(pgid, 0)
        except OSError:
            print("i  Process group was already terminated")
            return

        print(f"🛑 Stopping benchmark process group (PGID: {pgid})...")
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            return
        for _ in range(10):
            try:
                os.killpg(pgid, 0)
                time.sleep(0.5)
            except OSError:
                return
        print("⚡ Force killing process group...")
        with contextlib.suppress(OSError):
            os.killpg(pgid, signal.SIGKILL)

    @staticmethod
    def _kill_single_process(pid: int) -> None:
        """Fallback: terminate a single PID (for old status files w/o pgid)."""
        try:
            os.kill(pid, 0)
        except OSError:
            print("i  Process was already terminated")
            return
        print(f"🛑 Stopping benchmark process (PID: {pid})...")
        os.kill(pid, signal.SIGTERM)
        for _ in range(10):
            try:
                os.kill(pid, 0)
                time.sleep(0.5)
            except OSError:
                return
        print("⚡ Force killing...")
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated Background Benchmark Runner")
    parser.add_argument("--setup", action="store_true", help="Setup isolated environment")
    parser.add_argument(
        "--run",
        help=("Run benchmarks for specific commit(s). Accepts a single commit, range, or space-separated list."),
    )
    parser.add_argument("--bench", help="Filter to specific benchmark")
    parser.add_argument("--branch", help="Specify branch for benchmarking")
    parser.add_argument("--commit", help="Commit hash to benchmark")
    parser.add_argument("--status", action="store_true", help="Show benchmark status")
    parser.add_argument("--stop", action="store_true", help="Stop running benchmarks")
    parser.add_argument("--clean", action="store_true", help="Clean up benchmark environment")
    parser.add_argument("--force", action="store_true", help="Force clean without confirmation")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Force update isolated repository",
    )
    parser.add_argument("--view", action="store_true", help="View benchmark results")
    parser.add_argument("--quick", action="store_true", help="Run benchmarks in quick mode")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--working-dir", help="Working directory path")

    args = parser.parse_args()
    runner = IsolatedBenchmarkRunner(args.working_dir)

    if args.setup:
        runner.setup_isolated_repo()
        runner.remove_git_hooks()
        print("\n🎉 Isolated benchmark environment ready!")
        print("📝 Usage:")
        print(f"  python {__file__} --run HEAD --bench GKPlusTreeInsert")
        print(f"  python {__file__} --status")
        print(f"  python {__file__} --view")

    elif args.status:
        runner.status.show()

    elif args.stop:
        runner.stop_benchmarks()

    elif args.clean:
        runner.clean_benchmarks(force=args.force)

    elif args.update:
        ok = runner.force_update_repo()
        sys.exit(0 if ok else 1)

    elif args.view:
        runner.view_results()

    elif args.run:
        # Prevent overlapping runs (skip for internal worker invocations)
        if not args.worker:
            st = runner.status.get()
            if st.get("status") == "running":
                print(f"❌ Benchmark already in progress (PID: {st.get('pid')}). Stop it or wait.")
                sys.exit(1)

        commit = args.commit or args.run
        if not commit:
            print("❌ Must specify commit hash")
            sys.exit(1)

        if not runner.repo_dir.exists():
            print("🔧 Setting up environment first...")
            runner.setup_isolated_repo()

        if args.worker:
            runner.run_benchmarks_worker(commit, args.bench, args.branch, quick=args.quick)
            return

        runner.run_benchmarks_background(commit, args.bench, args.branch, quick=args.quick)
        time.sleep(2)
        runner.status.show()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
