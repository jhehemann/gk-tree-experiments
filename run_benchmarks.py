#!/usr/bin/env python3
"""
Benchmark runner script for G+ trees and K-lists.

This script provides a convenient interface for running various benchmark
scenarios with appropriate configurations for robust performance testing.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks for G+ trees and K-lists",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmarks.py --setup                    # Initial setup
  python run_benchmarks.py --quick                    # Quick development test
  python run_benchmarks.py --full                     # Full benchmark suite
  python run_benchmarks.py --klist --insert           # KList insert benchmarks
  python run_benchmarks.py --klist --retrieve         # KList retrieve benchmarks
  python run_benchmarks.py --advanced                 # Advanced KList scenarios
  python run_benchmarks.py --report                   # Generate HTML report
        """
    )
    
    # Setup options
    parser.add_argument('--setup', action='store_true',
                        help='Initialize ASV environment (run once)')
    
    # Benchmark scope options
    parser.add_argument('--quick', action='store_true',
                        help='Run quick development benchmarks')
    parser.add_argument('--full', action='store_true',
                        help='Run full benchmark suite')
    parser.add_argument('--klist', action='store_true',
                        help='Run KList benchmarks only')
    parser.add_argument('--advanced', action='store_true',
                        help='Run advanced KList scenario benchmarks only')
    
    # Operation-specific options
    parser.add_argument('--insert', action='store_true',
                        help='Run insert operation benchmarks only')
    parser.add_argument('--retrieve', action='store_true',
                        help='Run retrieve operation benchmarks only')
    parser.add_argument('--memory', action='store_true',
                        help='Run memory usage benchmarks only')
    
    # Reporting options
    parser.add_argument('--report', action='store_true',
                        help='Generate HTML report from existing results')
    parser.add_argument('--show', action='store_true',
                        help='Show latest results in terminal')
    
    # Advanced options
    parser.add_argument('--parallel', action='store_true',
                        help='Run benchmarks in parallel (if supported)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--machine', type=str,
                        help='Specify machine name for results')
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path('asv.conf.json').exists():
        print("❌ Error: asv.conf.json not found. Please run from project root.")
        return 1
    
    # Setup ASV environment
    if args.setup:
        print("🔧 Setting up ASV environment...")
        if not run_command(['poetry', 'run', 'asv', 'machine', '--yes'], 
                          "Configuring ASV machine info"):
            return 1
        print("✅ ASV setup complete!")
        return 0
    
    # Generate reports
    if args.report:
        print("📊 Generating HTML report...")
        if run_command(['poetry', 'run', 'asv', 'publish'], 
                      "Publishing results"):
            print("🌐 Starting web server for report...")
            run_command(['poetry', 'run', 'asv', 'preview'], 
                       "Opening report in browser")
        return 0
    
    # Show results
    if args.show:
        run_command(['poetry', 'run', 'asv', 'show'], 
                   "Showing latest results")
        return 0
    
    # Build benchmark command
    base_cmd = ['poetry', 'run', 'asv', 'run']
    
    # Add machine specification if provided
    if args.machine:
        base_cmd.extend(['--machine', args.machine])
    
    # Add parallel execution if requested
    if args.parallel:
        base_cmd.append('--parallel')
    
    # Add verbose flag if requested
    if args.verbose:
        base_cmd.append('--verbose')
    
    # Determine benchmark patterns based on arguments
    patterns = []
    
    if args.quick:
        patterns.append('--quick')
        description = "Quick development benchmarks"
    elif args.full:
        description = "Full benchmark suite"
    else:
        # Build specific benchmark patterns
        if args.klist:
            patterns.extend(['-b', 'benchmark_klist'])
        if args.advanced:
            patterns.extend(['-b', 'benchmark_gkplus_tree.*advanced'])
        
        # Add operation-specific filters
        operation_filters = []
        if args.insert:
            operation_filters.append('time_insert')
        if args.retrieve:
            operation_filters.append('time_retrieve')
        if args.memory:
            operation_filters.append('peakmem')
        
        for op_filter in operation_filters:
            patterns.extend(['-b', op_filter])
        
        # Default description
        if patterns:
            description = f"Targeted benchmarks: {' '.join(patterns)}"
        else:
            description = "All benchmarks"
    
    # Execute benchmark command
    cmd = base_cmd + patterns
    success = run_command(cmd, description)
    
    if success:
        print("\n✅ Benchmarks completed successfully!")
        print("\n📋 Next steps:")
        print("  • View results: python run_benchmarks.py --show")
        print("  • Generate report: python run_benchmarks.py --report")
        print("  • Compare with previous: poetry run asv compare HEAD HEAD~1")
    else:
        print("\n❌ Benchmarks failed!")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
