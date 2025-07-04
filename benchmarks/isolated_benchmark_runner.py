#!/usr/bin/env python3
"""
Isolated Background Benchmark Runner

This script implements benchmarking by:
1. Running benchmarks in a completely isolated directory
2. Working in the background without blocking development
3. Automatically triggering on commits to benchmark branches
4. Never interfering with the working directory

Features:
- Complete isolation from working directory
- Background execution with progress tracking
- Automatic git hook integration
- Local result storage and viewing
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Thread
import signal

class IsolatedBenchmarkRunner:
    def __init__(self, working_dir=None):
        self.working_dir = Path(working_dir or os.getcwd())
        self.benchmark_dir = self.working_dir.parent / ".isolated-benchmarks"
        self.repo_dir = self.benchmark_dir / "gplus-trees"
        self.results_dir = self.benchmark_dir / "results"
        self.logs_dir = self.benchmark_dir / "logs"
        self.status_file = self.benchmark_dir / "status.json"
        
        # Ensure directories exist
        self.benchmark_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
    
    def setup_isolated_repo(self):
        """Set up completely isolated repository for benchmarking."""
        print("🔧 Setting up isolated benchmark environment...")
        
        if self.repo_dir.exists():
            print("📡 Updating existing isolated repository...")
            self._run_command(["git", "fetch", "--all"], cwd=self.repo_dir)
            self._run_command(["git", "reset", "--hard", "origin/main"], cwd=self.repo_dir)
        else:
            print("📥 Cloning repository for isolated benchmarking...")
            # Get remote URL from working directory
            result = self._run_command(["git", "remote", "get-url", "origin"], cwd=self.working_dir)
            repo_url = result.stdout.strip()
            
            self._run_command(["git", "clone", repo_url, str(self.repo_dir)], cwd=self.benchmark_dir)
        
        # Create ASV configuration for isolated environment
        asv_config = {
            "version": 1,
            "project": "gplus-trees-isolated",
            "project_url": "isolated-benchmarking",
            "repo": str(self.repo_dir),
            "environment_type": "virtualenv",
            "benchmark_dir": "benchmarks",
            "env_dir": str(self.benchmark_dir / ".asv" / "env"),
            "results_dir": str(self.results_dir),
            "html_dir": str(self.benchmark_dir / "html"),
            "pythons": ["3.11"],
            "matrix": {"numpy": [""]},
            "dvcs": "git",
            "branches": ["performance-refactor", "main"]
        }
        
        with open(self.repo_dir / "asv.conf.json", "w") as f:
            json.dump(asv_config, f, indent=2)
        
        print("✅ Isolated environment ready!")
    
    def _run_command(self, cmd, cwd=None, capture_output=True):
        """Run a command safely."""
        try:
            result = subprocess.run(
                cmd, 
                cwd=cwd or self.benchmark_dir,
                capture_output=capture_output,
                text=True,
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {' '.join(cmd)}")
            print(f"Error: {e.stderr if e.stderr else e}")
            raise
    
    def update_status(self, status, message="", commit=None):
        """Update status file for progress tracking."""
        status_data = {
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "commit": commit,
            "pid": os.getpid()
        }
        
        with open(self.status_file, "w") as f:
            json.dump(status_data, f, indent=2)
    
    def get_status(self):
        """Get current benchmark status."""
        if not self.status_file.exists():
            return {"status": "idle", "message": "No benchmarks running"}
        
        try:
            with open(self.status_file, "r") as f:
                return json.load(f)
        except:
            return {"status": "unknown", "message": "Could not read status"}
    
    def run_benchmarks_background(self, commit_hash, benchmark_filter=None, branch=None):
        """Run benchmarks in background without blocking."""
        
        def benchmark_worker():
            try:
                self.update_status("running", f"Starting benchmarks for {commit_hash}", commit_hash)
                
                # Update isolated repo
                print("📡 Updating isolated repository...")
                self._run_command(["git", "fetch", "--all"], cwd=self.repo_dir)
                
                # Checkout the specific commit
                if branch:
                    self._run_command(["git", "checkout", branch], cwd=self.repo_dir)
                    self._run_command(["git", "reset", "--hard", f"origin/{branch}"], cwd=self.repo_dir)
                
                # Setup ASV environment
                self.update_status("running", "Setting up ASV environment...", commit_hash)
                self._run_command(["poetry", "run", "asv", "machine", "--yes"], cwd=self.repo_dir)
                
                # Run benchmarks
                bench_cmd = ["poetry", "run", "asv", "run", "--quick", "--python=3.11", commit_hash]
                if benchmark_filter:
                    bench_cmd.extend(["--bench", benchmark_filter])
                
                self.update_status("running", f"Running benchmarks for {commit_hash}...", commit_hash)
                
                # Create log file for this run
                log_file = self.logs_dir / f"benchmark_{commit_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                
                with open(log_file, "w") as f:
                    f.write(f"Benchmark run started at {datetime.now()}\n")
                    f.write(f"Commit: {commit_hash}\n")
                    f.write(f"Filter: {benchmark_filter or 'All benchmarks'}\n")
                    f.write("="*50 + "\n\n")
                    f.flush()
                    
                    result = subprocess.run(
                        bench_cmd,
                        cwd=self.repo_dir,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                
                if result.returncode == 0:
                    # Generate HTML report
                    self.update_status("running", "Generating HTML report...", commit_hash)
                    self._run_command(["poetry", "run", "asv", "publish"], cwd=self.repo_dir)
                    
                    self.update_status("completed", f"Benchmarks completed successfully for {commit_hash}", commit_hash)
                    print(f"✅ Benchmarks completed for {commit_hash}")
                    print(f"📊 Results: {self.benchmark_dir / 'html' / 'index.html'}")
                    print(f"📋 Log: {log_file}")
                else:
                    self.update_status("failed", f"Benchmarks failed for {commit_hash}", commit_hash)
                    print(f"❌ Benchmarks failed for {commit_hash}")
                    print(f"📋 Check log: {log_file}")
                
            except Exception as e:
                error_msg = f"Benchmark error: {str(e)}"
                self.update_status("failed", error_msg, commit_hash)
                print(f"❌ {error_msg}")
        
        # Start background thread
        thread = Thread(target=benchmark_worker, daemon=True)
        thread.start()
        
        print(f"🚀 Benchmarks started in background for commit {commit_hash}")
        print(f"📊 Monitor progress: python {__file__} --status")
        print(f"🔍 View results when done: python {__file__} --view")
        
        return thread
    
    def install_git_hooks(self):
        """Install git hooks for automatic benchmarking."""
        hooks_dir = self.working_dir / ".git" / "hooks"
        post_commit_hook = hooks_dir / "post-commit"
        
        hook_content = f"""#!/bin/bash
# Isolated benchmark automation hook
# Auto-triggers benchmarks for specific branches

BRANCH=$(git branch --show-current)
COMMIT=$(git rev-parse HEAD)
BENCHMARK_BRANCHES=("performance-refactor" "main")

# Check if current branch should trigger benchmarks
if [[ " ${{BENCHMARK_BRANCHES[@]}} " =~ " ${{BRANCH}} " ]]; then
    echo "🚀 Triggering isolated benchmarks for $BRANCH ($COMMIT)"
    
    # Run benchmarks in background (non-blocking)
    nohup python "{__file__}" --auto-run --commit "$COMMIT" --branch "$BRANCH" >/dev/null 2>&1 &
    
    echo "📊 Benchmarks started in background. Monitor with: python {__file__} --status"
fi
"""
        
        with open(post_commit_hook, "w") as f:
            f.write(hook_content)
        
        # Make executable
        os.chmod(post_commit_hook, 0o755)
        
        print(f"✅ Git hooks installed at {post_commit_hook}")
        print("🔄 Benchmarks will auto-run on commits to performance-refactor and main branches")
    
    def view_results(self):
        """Open benchmark results in browser."""
        html_file = self.benchmark_dir / "html" / "index.html"
        if html_file.exists():
            import webbrowser
            webbrowser.open(f"file://{html_file}")
            print(f"🌐 Opening results: {html_file}")
        else:
            print("❌ No results found. Run benchmarks first.")
    
    def show_status(self):
        """Show current benchmark status."""
        status = self.get_status()
        
        print(f"📊 Benchmark Status: {status['status'].upper()}")
        print(f"💬 Message: {status.get('message', 'No message')}")
        print(f"⏰ Last Update: {status.get('timestamp', 'Unknown')}")
        
        if status.get('commit'):
            print(f"🔍 Commit: {status['commit']}")
        
        # Show recent logs
        log_files = sorted(self.logs_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        if log_files:
            print(f"\n📋 Recent Logs:")
            for log_file in log_files[:3]:
                print(f"  - {log_file.name}")

def main():
    parser = argparse.ArgumentParser(description="Isolated Background Benchmark Runner")
    parser.add_argument("--setup", action="store_true", help="Setup isolated environment and git hooks")
    parser.add_argument("--run", help="Run benchmarks for specific commit")
    parser.add_argument("--bench", help="Filter to specific benchmark")
    parser.add_argument("--branch", help="Specify branch for benchmarking")
    parser.add_argument("--auto-run", action="store_true", help="Auto-run mode (used by git hooks)")
    parser.add_argument("--commit", help="Commit hash to benchmark")
    parser.add_argument("--status", action="store_true", help="Show benchmark status")
    parser.add_argument("--view", action="store_true", help="View benchmark results")
    parser.add_argument("--working-dir", help="Working directory path")
    
    args = parser.parse_args()
    
    runner = IsolatedBenchmarkRunner(args.working_dir)
    
    if args.setup:
        runner.setup_isolated_repo()
        runner.install_git_hooks()
        print("\n🎉 Isolated benchmark environment ready!")
        print("📝 Usage:")
        print(f"  - Run manually: python {__file__} --run HEAD --bench GKPlusTreeInsert")
        print(f"  - Check status: python {__file__} --status")
        print(f"  - View results: python {__file__} --view")
        print("  - Auto-runs on commits to benchmark branches!")
        
    elif args.status:
        runner.show_status()
        
    elif args.view:
        runner.view_results()
        
    elif args.run or args.auto_run:
        commit = args.commit or args.run
        if not commit:
            print("❌ Must specify commit hash")
            sys.exit(1)
        
        # Ensure environment is set up
        if not runner.repo_dir.exists():
            print("🔧 Setting up environment first...")
            runner.setup_isolated_repo()
        
        runner.run_benchmarks_background(commit, args.bench, args.branch)
        
        # If not auto-run, wait a bit to show initial status
        if not args.auto_run:
            time.sleep(2)
            runner.show_status()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
