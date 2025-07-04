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
        
        # Get current branch from working directory
        current_branch_result = self._run_command(["git", "branch", "--show-current"], cwd=self.working_dir)
        current_branch = current_branch_result.stdout.strip()
        print(f"📍 Current branch: {current_branch}")
        
        if self.repo_dir.exists():
            print("📡 Updating existing isolated repository...")
            self._run_command(["git", "fetch", "--all"], cwd=self.repo_dir)
            # Check out the current branch instead of main
            self._run_command(["git", "checkout", current_branch], cwd=self.repo_dir)
            self._run_command(["git", "reset", "--hard", f"origin/{current_branch}"], cwd=self.repo_dir)
        else:
            print("📥 Cloning repository for isolated benchmarking...")
            # Get remote URL from working directory
            result = self._run_command(["git", "remote", "get-url", "origin"], cwd=self.working_dir)
            repo_url = result.stdout.strip()
            
            self._run_command(["git", "clone", "-b", current_branch, repo_url, str(self.repo_dir)], cwd=self.benchmark_dir)
        
        # Ensure all benchmark branches exist locally for ASV
        print("🔄 Setting up local benchmark branches...")
        benchmark_branches = ["main", "performance-refactor"]
        for branch in benchmark_branches:
            try:
                # Check if local branch exists
                self._run_command(["git", "show-ref", "--verify", f"refs/heads/{branch}"], cwd=self.repo_dir)
                print(f"✅ Local branch '{branch}' already exists")
            except subprocess.CalledProcessError:
                # Branch doesn't exist locally, create it from remote
                try:
                    self._run_command(["git", "checkout", "-b", branch, f"origin/{branch}"], cwd=self.repo_dir)
                    print(f"✅ Created local branch '{branch}' from origin/{branch}")
                except subprocess.CalledProcessError:
                    print(f"⚠️  Could not create local branch '{branch}' - remote may not exist")
        
        # Switch back to the current branch
        self._run_command(["git", "checkout", current_branch], cwd=self.repo_dir)
        
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
    
    def _resolve_commit_reference(self, commit_ref, branch=None):
        """Resolve commit references like HEAD^!, branch^! to actual commit hashes."""
        # Handle special syntax like HEAD^!, performance-refactor^!
        if commit_ref.endswith("^!"):
            base_ref = commit_ref[:-2]  # Remove ^!
            
            # If it's HEAD^!, resolve from working directory
            if base_ref == "HEAD":
                result = self._run_command(["git", "rev-parse", "HEAD"], cwd=self.working_dir)
                resolved_commit = result.stdout.strip()
                print(f"🔍 Resolved {commit_ref} to {resolved_commit[:8]} (current branch HEAD)")
                return resolved_commit
            
            # If it's branch^!, resolve from that branch
            elif branch or base_ref != "HEAD":
                target_branch = branch or base_ref
                try:
                    result = self._run_command(["git", "rev-parse", f"origin/{target_branch}"], cwd=self.working_dir)
                    resolved_commit = result.stdout.strip()
                    print(f"🔍 Resolved {commit_ref} to {resolved_commit[:8]} (latest on {target_branch})")
                    return resolved_commit
                except subprocess.CalledProcessError:
                    print(f"⚠️  Could not resolve {commit_ref}, using as-is")
                    return commit_ref
        
        # For regular commit references (hash, HEAD, branch names), return as-is
        return commit_ref
    
    def update_status(self, status, message="", commit=None, subprocess_pid=None):
        """Update status file for progress tracking."""
        status_data = {
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "commit": commit,
            "pid": subprocess_pid or os.getpid()
        }
        
        with open(self.status_file, "w") as f:
            json.dump(status_data, f, indent=2)
    
    def get_status(self):
        """Get current benchmark status."""
        if not self.status_file.exists():
            return {"status": "idle", "message": "No benchmarks running"}
        
        try:
            with open(self.status_file, "r") as f:
                status = json.load(f)
            
            # Check if the process is still running
            if status.get("status") == "running" and "pid" in status:
                try:
                    os.kill(status["pid"], 0)  # Check if process exists
                except OSError:
                    # Process is dead, update status
                    status["status"] = "failed"
                    status["message"] = "Process terminated unexpectedly"
                    with open(self.status_file, "w") as f:
                        json.dump(status, f, indent=2)
            
            return status
        except:
            return {"status": "unknown", "message": "Could not read status"}
    
    def run_benchmarks_background(self, commit_hash, benchmark_filter=None, branch=None):
        """Run benchmarks in background without blocking."""
        
        def benchmark_worker():
            try:
                # Handle special commit syntax
                resolved_commit = self._resolve_commit_reference(commit_hash, branch)
                
                self.update_status("running", f"Starting benchmarks for {resolved_commit}", resolved_commit)
                
                # Update isolated repo
                print("📡 Updating isolated repository...")
                self._run_command(["git", "fetch", "--all"], cwd=self.repo_dir)
                
                # Checkout the specific commit or branch
                if branch:
                    print(f"🔄 Switching to branch: {branch}")
                    self._run_command(["git", "checkout", branch], cwd=self.repo_dir)
                    self._run_command(["git", "reset", "--hard", f"origin/{branch}"], cwd=self.repo_dir)
                else:
                    # If no branch specified, checkout the commit directly
                    print(f"🔄 Checking out commit: {resolved_commit}")
                    self._run_command(["git", "checkout", resolved_commit], cwd=self.repo_dir)
                
                # Setup ASV environment
                self.update_status("running", "Setting up ASV environment...", resolved_commit)
                self._run_command(["poetry", "run", "asv", "machine", "--yes"], cwd=self.repo_dir)
                
                # Run benchmarks
                # Only use commit^! format if the original commit_hash ended with ^!
                if commit_hash.endswith("^!"):
                    commit_spec = f"{resolved_commit}^!"
                else:
                    commit_spec = resolved_commit
                bench_cmd = ["poetry", "run", "asv", "run", "--quick", "--python=3.11", commit_spec]
                if benchmark_filter:
                    bench_cmd.extend(["--bench", benchmark_filter])
                
                self.update_status("running", f"Running benchmarks for {resolved_commit}...", resolved_commit)
                
                # Create log file for this run
                short_commit = resolved_commit[:8]
                
                # Get current branch name for log file
                try:
                    branch_result = self._run_command(["git", "branch", "--show-current"], cwd=self.repo_dir)
                    current_branch = branch_result.stdout.strip()
                    if not current_branch:  # Detached HEAD
                        current_branch = "detached"
                except:
                    current_branch = "unknown"
                
                log_file = self.logs_dir / f"benchmark_{current_branch}_{short_commit}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                
                with open(log_file, "w") as f:
                    f.write(f"Benchmark run started at {datetime.now()}\n")
                    f.write(f"Commit: {resolved_commit}\n")
                    f.write(f"Filter: {benchmark_filter or 'All benchmarks'}\n")
                    f.write("="*50 + "\n\n")
                    f.flush()
                    
                    # Start ASV process and track its PID
                    process = subprocess.Popen(
                        bench_cmd,
                        cwd=self.repo_dir,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                    
                    # Update status with subprocess PID
                    self.update_status("running", f"Running benchmarks for {resolved_commit}...", resolved_commit, process.pid)
                    
                    # Wait for completion
                    result = process.wait()
                
                if result == 0:
                    # Generate HTML report
                    self.update_status("running", "Generating HTML report...", resolved_commit)
                    self._run_command(["poetry", "run", "asv", "publish"], cwd=self.repo_dir)
                    
                    self.update_status("completed", f"Benchmarks completed successfully for {resolved_commit}", resolved_commit)
                    print(f"✅ Benchmarks completed for {resolved_commit}")
                    print(f"📊 Results: {self.benchmark_dir / 'html' / 'index.html'}")
                    print(f"📋 Log: {log_file}")
                else:
                    self.update_status("failed", f"Benchmarks failed for {resolved_commit}", resolved_commit)
                    print(f"❌ Benchmarks failed for {resolved_commit}")
                    print(f"📋 Check log: {log_file}")
                
            except Exception as e:
                error_msg = f"Benchmark error: {str(e)}"
                self.update_status("failed", error_msg, resolved_commit if 'resolved_commit' in locals() else commit_hash)
                print(f"❌ {error_msg}")
        
        # Start background thread
        thread = Thread(target=benchmark_worker, daemon=True)
        thread.start()
        
        # Show resolved commit if it was a special reference
        if commit_hash.endswith("^!"):
            resolved = self._resolve_commit_reference(commit_hash, branch)
            print(f"🚀 Benchmarks started in background for commit {commit_hash} → {resolved[:8]}")
        else:
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
        """Open benchmark results in browser using local web server."""
        # Check if we have any results
        if not (self.results_dir / "benchmarks.json").exists():
            print("❌ No results found. Run benchmarks first.")
            return
            
        html_dir = self.benchmark_dir / "html"
        
        # If HTML doesn't exist, generate it
        if not html_dir.exists() or not (html_dir / "index.html").exists():
            print("📊 Generating HTML results...")
            try:
                self._run_command(["poetry", "run", "asv", "publish"], cwd=self.repo_dir)
                print("✅ HTML results generated successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to generate HTML results: {e}")
                print("💡 Try running: ./benchmark setup to ensure proper environment")
                return
        
        # Use ASV preview to start local web server and open browser
        print("🌐 Starting local web server and opening results in browser...")
        print("📝 Press Ctrl+C to stop the web server")
        try:
            self._run_command(
                ["poetry", "run", "asv", "preview", "--browser", "--html-dir", str(html_dir)], 
                cwd=self.repo_dir
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start preview server: {e}")
            print("💡 Falling back to opening HTML file directly...")
            # Fallback to direct file opening
            html_file = html_dir / "index.html"
            if html_file.exists():
                import webbrowser
                webbrowser.open(f"file://{html_file}")
                print(f"📂 Opened: {html_file}")
                print("⚠️  Note: Some features may not work due to browser security restrictions")
        except KeyboardInterrupt:
            print("\n🛑 Web server stopped")
    
    def stop_benchmarks(self):
        """Gracefully stop running benchmarks."""
        status = self.get_status()
        
        if status["status"] != "running":
            print(f"ℹ️  No benchmarks currently running (status: {status['status']})")
            return
        
        pid = status.get("pid")
        if not pid:
            print("❌ No process ID found in status")
            return
        
        try:
            # Check if process exists
            os.kill(pid, 0)
            print(f"🛑 Stopping benchmark process (PID: {pid})...")
            
            # Try graceful termination first (SIGTERM)
            os.kill(pid, signal.SIGTERM)
            
            # Wait a bit to see if it terminates gracefully
            import time
            for i in range(10):
                try:
                    os.kill(pid, 0)  # Check if still running
                    time.sleep(0.5)
                except OSError:
                    # Process has terminated
                    break
            else:
                # Process still running after 5 seconds, force kill
                print("⚡ Process did not terminate gracefully, forcing termination...")
                os.kill(pid, signal.SIGKILL)
            
            # Update status
            self.update_status("stopped", "Benchmark stopped by user", status.get("commit"))
            print("✅ Benchmark process stopped successfully")
            
        except OSError:
            print("ℹ️  Process was already terminated")
            self.update_status("stopped", "Process was already terminated", status.get("commit"))

    def clean_benchmarks(self, force=False):
        """Clean up benchmark environment with confirmation."""
        if not force:
            print("⚠️  This will permanently delete:")
            print("   • All benchmark results and data")
            print("   • Generated HTML reports")
            print("   • Benchmark execution logs")
            print("   • Isolated repository clone")
            print("   • Git hooks")
            print("After cleanup, you will need to re-run setup to create a new environment.")
            print()
            
            response = input("Are you sure you want to continue? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("❌ Operation cancelled")
                return False
        
        print("🧹 Cleaning up isolated benchmark environment...")
        
        # Remove the entire isolated benchmarks directory
        if self.benchmark_dir.exists():
            shutil.rmtree(self.benchmark_dir)
            print(f"✅ Cleaned up {self.benchmark_dir}")
        else:
            print("ℹ️  No benchmark directory to clean")
        
        # Remove git hooks
        hooks_dir = self.working_dir / ".git" / "hooks"
        post_commit_hook = hooks_dir / "post-commit"
        
        if post_commit_hook.exists():
            post_commit_hook.unlink()
            print("✅ Removed git hooks")
        else:
            print("ℹ️  No git hooks to remove")
        
        return True

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
        
        # If benchmarks are running, show recent log output for progress
        if status.get('status') == 'running' and log_files:
            recent_log = log_files[0]
            print(f"\n🖥️  Recent output from {recent_log.name}:")
            try:
                with open(recent_log, 'r') as f:
                    tail_lines = f.readlines()[-5:]
                for line in tail_lines:
                    print(f"  {line.rstrip()}")
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser(description="Isolated Background Benchmark Runner")
    parser.add_argument("--setup", action="store_true", help="Setup isolated environment and git hooks")
    parser.add_argument("--run", help="Run benchmarks for specific commit")
    parser.add_argument("--bench", help="Filter to specific benchmark")
    parser.add_argument("--branch", help="Specify branch for benchmarking")
    parser.add_argument("--auto-run", action="store_true", help="Auto-run mode (used by git hooks)")
    parser.add_argument("--commit", help="Commit hash to benchmark")
    parser.add_argument("--status", action="store_true", help="Show benchmark status")
    parser.add_argument("--stop", action="store_true", help="Stop running benchmarks")
    parser.add_argument("--clean", action="store_true", help="Clean up benchmark environment")
    parser.add_argument("--force", action="store_true", help="Force clean without confirmation")
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
        
    elif args.stop:
        runner.stop_benchmarks()
        
    elif args.clean:
        runner.clean_benchmarks(force=args.force)
        
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
