#!/usr/bin/env python3
"""
Isolated Background Benchmark Runner

This script implements benchmarking by:
1. Running benchmarks in a completely isolated directory
2. Working in the background without blocking development
3. Never interfering with the working directory

Features:
- Complete isolation from working directory
- Background execution with progress tracking
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
        
        # Discover and set up all relevant benchmark branches
        print("🔄 Setting up local benchmark branches...")
        
        # Get list of remote branches that might be relevant for benchmarking
        remote_branches_result = self._run_command(["git", "branch", "-r"], cwd=self.repo_dir)
        remote_branches = []
        for line in remote_branches_result.stdout.strip().split('\n'):
            branch = line.strip()
            if '->' not in branch and branch.startswith('origin/'):
                branch_name = branch.replace('origin/', '')
                if branch_name not in ['HEAD', 'main', 'develop']:  # Skip meta branches
                    remote_branches.append(branch_name)
        
        # Include current branch and commonly used benchmark branches
        benchmark_branches = list(set([current_branch] + remote_branches))
        print(f"🔍 Found branches for benchmarking: {', '.join(sorted(benchmark_branches))}")
        
        # Ensure local branches exist for all benchmark branches
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
            "branches": benchmark_branches
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
        """Resolve commit references, with special handling for branch^! syntax."""
        
        # Handle special branch^! syntax (ASV doesn't understand origin/branch^!)
        if commit_ref.endswith("^!"):
            base_ref = commit_ref[:-2]  # Remove ^!
            
            # If it's HEAD^!, let ASV handle it as-is
            if base_ref == "HEAD":
                print(f"🔍 Using {commit_ref} as-is for ASV")
                return commit_ref, None
            
            # Try to resolve as a commit hash first
            try:
                result = self._run_command(["git", "rev-parse", base_ref], cwd=self.working_dir)
                resolved_commit = result.stdout.strip()
                resolved_ref = f"{resolved_commit}^!"
                print(f"🔍 Resolved {commit_ref} to {resolved_ref[:12]}...")
                return resolved_ref, None
            except subprocess.CalledProcessError:
                # Not a commit hash, try as branch name
                try:
                    result = self._run_command(["git", "rev-parse", f"origin/{base_ref}"], cwd=self.working_dir)
                    resolved_commit = result.stdout.strip()
                    resolved_ref = f"{resolved_commit}^!"
                    print(f"🔍 Resolved {commit_ref} to {resolved_ref[:12]}...")
                    return resolved_ref, base_ref
                except subprocess.CalledProcessError:
                    print(f"⚠️  Could not resolve {commit_ref}, using base reference: {base_ref}")
                    return base_ref, None
        
        # For all other cases (ranges, hashes, etc.), let ASV handle it natively
        print(f"🔍 Passing {commit_ref} directly to ASV")
        return commit_ref, None
    
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
            
            # Check if the process is still running (only if status is currently "running")
            if status.get("status") == "running" and "pid" in status:
                try:
                    os.kill(status["pid"], 0)  # Check if process exists
                except OSError:
                    # Process is dead - check if it completed successfully by examining log
                    if self._check_benchmark_completion(status.get("timestamp", "")):
                        status["status"] = "completed"
                        status["message"] = f"Benchmarks completed successfully for {status.get('commit', 'unknown')}"
                    else:
                        status["status"] = "failed"
                        status["message"] = "Process terminated unexpectedly"
                    
                    with open(self.status_file, "w") as f:
                        json.dump(status, f, indent=2)
            
            return status
        except:
            return {"status": "unknown", "message": "Could not read status"}
    
    def _check_benchmark_completion(self, start_timestamp):
        """Check if benchmark completed successfully by examining log files."""
        try:
            # Find the most recent log file that matches the timestamp
            log_files = sorted(self.logs_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
            
            if not log_files:
                return False
            
            recent_log = log_files[0]
            
            # Read all lines and find the last non-empty line
            with open(recent_log, 'r') as f:
                lines = [line.rstrip() for line in f if line.strip()]
            
            if not lines:
                return False
            
            last_line = lines[-1]
            # Check if the last line contains only '=' and whitespace
            if all(c == '=' or c.isspace() for c in last_line) and any(c == '=' for c in last_line):
                return True
            
            return False
            
        except Exception:
            return False
            
    def run_benchmarks_background(self, commit_hash, benchmark_filter=None, branch=None, quick=False):
        """Run benchmarks in background without blocking."""
        
        def benchmark_worker():
            try:
                # Handle special commit syntax (mainly for branch^! resolution)
                resolved_commit, resolved_branch = self._resolve_commit_reference(commit_hash, branch)
                effective_branch = branch or resolved_branch
                
                self.update_status("running", f"Starting benchmarks for {commit_hash}", commit_hash)
                
                # Update isolated repo
                print("📡 Updating isolated repository...")
                self._run_command(["git", "fetch", "--all"], cwd=self.repo_dir)
                
                # For branch-specific commits, ensure the branch exists locally
                if effective_branch:
                    print(f"🔄 Ensuring branch {effective_branch} exists locally...")
                    try:
                        self._run_command(["git", "checkout", effective_branch], cwd=self.repo_dir)
                        self._run_command(["git", "reset", "--hard", f"origin/{effective_branch}"], cwd=self.repo_dir)
                        # Update ASV config to include this branch
                        self._update_asv_config_for_branch(effective_branch)
                    except subprocess.CalledProcessError:
                        print(f"⚠️  Could not setup branch {effective_branch}, continuing anyway...")
                
                # Setup ASV environment (only once)
                self.update_status("running", "Setting up ASV environment...", commit_hash)
                self._run_command(["poetry", "run", "asv", "machine", "--yes"], cwd=self.repo_dir)
                
                # Let ASV handle the commit/range resolution and checkouts
                bench_cmd = ["poetry", "run", "asv", "run"]
                if quick:
                    bench_cmd.append("--quick")
                bench_cmd.extend(["--python=3.11", resolved_commit])
                if benchmark_filter:
                    bench_cmd.extend(["--bench", benchmark_filter])
                
                self.update_status("running", f"Running benchmarks for {commit_hash}...", commit_hash)
                
                # Create log file for this run  
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_commit = commit_hash.replace('..', '_to_').replace('^!', '_head').replace('/', '_')
                log_file = self.logs_dir / f"benchmark_{safe_commit}_{timestamp}.log"
                
                with open(log_file, "w") as f:
                    f.write(f"Benchmark run started at {datetime.now()}\n")
                    f.write(f"Original commit reference: {commit_hash}\n")
                    f.write(f"Resolved commit reference: {resolved_commit}\n")
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
                    self.update_status("running", f"Running benchmarks for {commit_hash}...", commit_hash, process.pid)
                    
                    # Wait for completion
                    result = process.wait()
                
                if result == 0:
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
        
        # Show appropriate message based on commit reference type
        if ".." in commit_hash:
            print(f"🚀 Benchmarks started in background for commit range: {commit_hash}")
        else:
            print(f"🚀 Benchmarks started in background for commit: {commit_hash}")
        
        print(f"📊 Monitor progress: python {__file__} --status")
        print(f"🔍 View results when done: python {__file__} --view")
        
        return thread
    
    def remove_git_hooks(self):
        """Remove git hooks to disable automatic benchmarking."""
        hooks_dir = self.working_dir / ".git" / "hooks"
        post_commit_hook = hooks_dir / "post-commit"
        
        # Remove any existing post-commit hook to disable auto benchmarks
        if post_commit_hook.exists():
            post_commit_hook.unlink()
            print("🗑️  Removed existing post-commit git hook (auto benchmarks on commit disabled)")
        else:
            print("ℹ️  No post-commit git hook to remove (auto benchmarks on commit already disabled)")
    
    def view_results(self):
        """Open benchmark results in browser using local web server."""
        # Check if we have any results
        if not (self.results_dir / "benchmarks.json").exists():
            print("❌ No results found. Run benchmarks first.")
            return
            
        html_dir = self.benchmark_dir / "html"
        
        # Always ensure we include all branches when generating/viewing HTML
        print("📊 Ensuring HTML includes all available branch results...")
        try:
            self._ensure_all_branches_in_html_generation()
            
            # Generate or update HTML
            if not html_dir.exists() or not (html_dir / "index.html").exists():
                print("📊 Generating HTML results...")
            else:
                print("📊 Updating HTML results...")
                
            self._run_command(["poetry", "run", "asv", "publish"], cwd=self.repo_dir)
            print("✅ HTML results generated successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to generate HTML results: {e}")
            print("💡 Try running: ./benchmark setup to ensure proper environment")
            return
        
        # Use ASV preview to start local web server and open browser at port 8081
        print("🌐 Starting local web server at port 8081 and opening results in browser...")
        print("📝 Press Ctrl+C to stop the web server")
        try:
            self._run_command(
            ["poetry", "run", "asv", "preview", "--browser", "--port", "8081", "--html-dir", str(html_dir)],
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

    def _update_asv_config_for_branch(self, branch_name):
        """Update ASV configuration to include the specified branch if not already present."""
        asv_config_path = self.repo_dir / "asv.conf.json"
        
        if not asv_config_path.exists():
            print("⚠️  ASV config not found, skipping branch update")
            return
        
        try:
            with open(asv_config_path, 'r') as f:
                asv_config = json.load(f)
            
            current_branches = asv_config.get("branches", [])
            
            if branch_name not in current_branches:
                current_branches.append(branch_name)
                asv_config["branches"] = current_branches
                
                with open(asv_config_path, 'w') as f:
                    json.dump(asv_config, f, indent=2)
                
                print(f"✅ Added '{branch_name}' to ASV branches configuration")
            else:
                print(f"✅ Branch '{branch_name}' already in ASV configuration")
        
        except Exception as e:
            print(f"⚠️  Could not update ASV config: {e}")
    
    def _ensure_all_branches_in_html_generation(self):
        """Ensure ASV includes results from all branches when generating HTML."""
        print("🔍 Ensuring all branches are included in HTML generation...")
        
        # Get all branches that have results by checking the results files
        results_files = list(self.results_dir.glob("MacBookPro.fritz.box/*.json"))
        if not results_files:
            print("⚠️  No machine-specific results found")
            return
        
        # Read all commit hashes from result files 
        branches_with_results = set()
        commit_hashes = set()
        
        for result_file in results_files:
            if result_file.name == "machine.json":
                continue
                
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                    commit_hash = result_data.get("commit_hash")
                    if commit_hash:
                        commit_hashes.add(commit_hash)
                        
                        # Try to find which branch this commit belongs to
                        try:
                            branch_result = self._run_command(
                                ["git", "branch", "--contains", commit_hash], 
                                cwd=self.repo_dir
                            )
                            # Parse branch names from output (remove * and whitespace)
                            branch_lines = [line.strip().lstrip('* ') for line in branch_result.stdout.strip().split('\n')]
                            for branch in branch_lines:
                                if branch and not branch.startswith('('):  # Skip detached HEAD
                                    branches_with_results.add(branch)
                                    print(f"🔍 Found branch '{branch}' for commit {commit_hash[:8]}")
                        except subprocess.CalledProcessError:
                            print(f"⚠️  Could not find branch for commit {commit_hash[:8]}")
            except (json.JSONDecodeError, OSError) as e:
                print(f"⚠️  Could not read {result_file.name}: {e}")
        
        print(f"📊 Found results for {len(commit_hashes)} commits across {len(branches_with_results)} branches: {', '.join(sorted(branches_with_results))}")
        
        if branches_with_results:
            # Update ASV config to include all branches with results
            self._update_asv_config_with_all_branches(branches_with_results)
            
            # Force ASV to rebuild its internal data structures to include all branches
            print("🔄 Rebuilding ASV data to include all branches...")
            try:
                # Try regular publish since --no-build is not supported in this ASV version
                self._run_command(["poetry", "run", "asv", "publish"], cwd=self.repo_dir)
                print("✅ HTML successfully regenerated with all branches")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Failed to publish HTML: {e}")
        else:
            print("⚠️  No branches found with results")

    def _update_asv_config_with_all_branches(self, branches_with_results):
        """Update ASV configuration to include all branches that have results."""
        asv_config_path = self.repo_dir / "asv.conf.json"
        
        if not asv_config_path.exists():
            print("⚠️  ASV config not found, skipping all-branches update")
            return
        
        try:
            with open(asv_config_path, 'r') as f:
                asv_config = json.load(f)
            
            current_branches = set(asv_config.get("branches", []))
            all_branches = current_branches.union(branches_with_results)
            
            if all_branches != current_branches:
                asv_config["branches"] = sorted(list(all_branches))
                
                with open(asv_config_path, 'w') as f:
                    json.dump(asv_config, f, indent=2)
                
                added_branches = all_branches - current_branches
                print(f"✅ Updated ASV config to include {len(added_branches)} additional branches: {', '.join(sorted(added_branches))}")
            else:
                print("✅ ASV config already includes all branches with results")
        
        except Exception as e:
            print(f"⚠️  Could not update ASV config: {e}")

def main():
    parser = argparse.ArgumentParser(description="Isolated Background Benchmark Runner")
    parser.add_argument("--setup", action="store_true", help="Setup isolated environment")
    parser.add_argument("--run", help="Run benchmarks for specific commit")
    parser.add_argument("--bench", help="Filter to specific benchmark")
    parser.add_argument("--branch", help="Specify branch for benchmarking")
    parser.add_argument("--commit", help="Commit hash to benchmark")
    parser.add_argument("--status", action="store_true", help="Show benchmark status")
    parser.add_argument("--stop", action="store_true", help="Stop running benchmarks")
    parser.add_argument("--clean", action="store_true", help="Clean up benchmark environment")
    parser.add_argument("--force", action="store_true", help="Force clean without confirmation")
    parser.add_argument("--view", action="store_true", help="View benchmark results")
    parser.add_argument("--quick", action="store_true", help="Run benchmarks in quick mode")
    parser.add_argument("--working-dir", help="Working directory path")
    
    args = parser.parse_args()
    
    runner = IsolatedBenchmarkRunner(args.working_dir)
    
    if args.setup:
        runner.setup_isolated_repo()
        runner.remove_git_hooks()
        print("\n🎉 Isolated benchmark environment ready!")
        print("📝 Usage:")
        print(f"  - Run manually: python {__file__} --run HEAD --bench GKPlusTreeInsert")
        print(f"  - Check status: python {__file__} --status")
        print(f"  - View results: python {__file__} --view")
        
    elif args.status:
        runner.show_status()
        
    elif args.stop:
        runner.stop_benchmarks()
        
    elif args.clean:
        runner.clean_benchmarks(force=args.force)
        
    elif args.view:
        runner.view_results()
        
    elif args.run:
        # Prevent overlapping benchmark executions
        current_status = runner.get_status()
        if current_status.get("status") == "running":
            print(f"❌ A benchmark is already in progress (PID: {current_status.get('pid')}). Please stop it or wait until it completes.")
            sys.exit(1)
        commit = args.commit or args.run
        if not commit:
            print("❌ Must specify commit hash")
            sys.exit(1)

        # Ensure environment is set up
        if not runner.repo_dir.exists():
            print("🔧 Setting up environment first...")
            runner.setup_isolated_repo()

        # Run benchmarks, pass quick flag
        runner.run_benchmarks_background(commit, args.bench, args.branch, quick=args.quick)

        # Wait a bit to show initial status
        time.sleep(2)
        runner.show_status()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
