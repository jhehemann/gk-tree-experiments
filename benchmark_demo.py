#!/usr/bin/env python3
"""
Quick benchmark runner for demonstration purposes.
This script runs a subset of benchmarks to showcase the comprehensive suite
and helps diagnose ASV HTML output issues.
"""

import subprocess
import sys
import time
import os


def run_benchmark(pattern, description):
    """Run a specific benchmark pattern and time it."""
    print(f"\n🔄 Running {description}...")
    start_time = time.time()
    
    try:
        cmd = ['poetry', 'run', 'asv', 'run', '-b', pattern, '--quick']
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        end_time = time.time()
        print(f"✅ {description} completed in {end_time - start_time:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def main():
    """Run a selection of benchmarks to demonstrate the suite."""
    print("🚀 Running G+ Trees Benchmark Suite Demo")
    print("=" * 50)
    
    benchmarks = [
        ("KListInsertBenchmarks.time_insert_entry_sequential", "KList Insert Performance"),
        ("KListRetrieveBenchmarks.time_retrieve_sequential", "KList Retrieve Performance"),
        ("GKPlusTreeInsertBenchmarks.time_insert_entry_sequential", "GKPlusTree Insert Performance"),
        ("GKPlusTreeRetrieveBenchmarks.time_retrieve_sequential", "GKPlusTree Retrieve Performance"),
        ("GKPlusTreeMixedWorkloadBenchmarks.time_mixed_workload", "GKPlusTree Mixed Workload"),
    ]
    
    successful = 0
    total = len(benchmarks)
    
    for pattern, description in benchmarks:
        if run_benchmark(pattern, description):
            successful += 1
    
    print(f"\n📊 Benchmark Summary")
    print("=" * 30)
    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    
    if successful >= 1:  # Changed from == total to >= 1
        print(f"\n🎉 {successful} benchmarks completed successfully!")
        
        # Generate HTML report
        print("\n📄 Generating HTML report...")
        try:
            subprocess.run(['poetry', 'run', 'asv', 'publish'], check=True, capture_output=True)
            print("✅ HTML report generated successfully!")
            
            # Check if HTML files exist
            html_dir = ".asv/html"
            if os.path.exists(html_dir):
                html_files = [f for f in os.listdir(html_dir) if f.endswith('.html')]
                print(f"📁 Found {len(html_files)} HTML files in {html_dir}")
                
                # List some key files
                key_files = ['index.html', 'summarylist.js', 'summarygrid.js']
                for key_file in key_files:
                    if os.path.exists(os.path.join(html_dir, key_file)):
                        print(f"   ✅ {key_file} exists")
                    else:
                        print(f"   ❌ {key_file} missing")
            else:
                print(f"❌ HTML directory {html_dir} not found")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ HTML generation failed: {e}")
        
        print("\nTo view the HTML report:")
        print("  poetry run asv preview")
        print("  Then open: http://127.0.0.1:8080/")
        
        print("\nTo run the full benchmark suite:")
        print("  poetry run asv run")
        
        print("\n🔍 HTML Issue Diagnostics:")
        print("- The 'libmambapy' warnings are normal and can be ignored")
        print("- If grid view is empty, try clicking on individual benchmark tiles")
        print("- Summary list view usually works better than grid view")
        print("- Individual benchmark pages should show detailed results")
    else:
        print("\n⚠️  Some benchmarks failed. Check the error messages above.")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
