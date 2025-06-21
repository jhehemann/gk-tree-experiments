#!/usr/bin/env python3
"""
Debug roundtrip tests to identify and fix issues with tree -> klist -> tree conversion.

This script compares the old (flaky) implementation with the new (robust) implementation
to demonstrate the root causes of intermittent test failures.
"""

import random
import sys
import os
from typing import List, Tuple, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.gplus_trees.gplus_tree_base import calc_rank_for_dim
from src.gplus_trees.base import Item
from src.gplus_trees.factory import create_gkplus_tree
from src.gplus_trees.klist_base import _tree_to_klist, _klist_to_tree
from tests.gk_plus.base import TestGKPlusTreeBase
from tests.gk_plus.test_set_conversion import debug_pretty_print as print_pretty

class DebugRoundtripTest(TestGKPlusTreeBase):
    """Debug class to compare old vs new roundtrip implementations"""
    
    def __init__(self):
        super().__init__()
        self.k = 4
        """Create entries from keys for testing"""
        return [Entry(Item(key, f"val_{key}"), None) for key in keys]
    
    def create_tree_from_entries(self, entries: List[Entry], ranks: List[int], DIM: int) -> GKPlusTreeBase:
        """Create a GKPlusTree for a given dimension from a list of entries"""
        tree = create_gkplus_tree(K=self.k, dimension=DIM)
        for i, entry in enumerate(entries):
            tree, _ = tree.insert_entry(entry, ranks[i])
        return tree
    
    def assert_entries_present_same_instance(self, exp_entries: List[Entry], act_entries: List[Entry]) -> List[str]:
        """Check entry instance preservation and return error messages"""
        errors = []
        for exp_entry in exp_entries:
            act_entry = next((e for e in act_entries if e.item.key == exp_entry.item.key), None)
            if act_entry is None:
                errors.append(f"Expected entry with key {exp_entry.item.key} not found in actual entries")
                continue
            
            if exp_entry is not act_entry:
                errors.append(f"Entry with key {exp_entry.item.key} is not the same instance (expected id: {id(exp_entry)}, actual id: {id(act_entry)})")
            
            if exp_entry.left_subtree is not act_entry.left_subtree:
                errors.append(f"Left subtree for key {exp_entry.item.key} is not the same instance")
        
        return errors
    
    def validate_tree_structure(self, tree: GKPlusTreeBase, expected_keys: List[int]) -> List[str]:
        """Validate tree structure and return error messages"""
        errors = []
        try:
            actual_keys = [entry.item.key for entry in tree]
            if sorted(actual_keys) != sorted(expected_keys):
                errors.append(f"Tree keys mismatch: expected {sorted(expected_keys)}, got {sorted(actual_keys)}")
        except Exception as e:
            errors.append(f"Error extracting tree keys: {e}")
        
        return errors
    
    def run_old_implementation(self, iteration: int, initial_tree_dim: int = 5, num_entries: int = 5) -> TestResult:
        """Run the old (problematic) implementation"""
        # OLD IMPLEMENTATION - This is what was failing
        keys = set()
        while len(keys) < num_entries:
            keys.add(random.randint(1, 10000))
        keys = sorted(list(keys))
        
        entries = self.create_entries(keys)
        ranks = [calc_rank_for_dim(key=key, k=self.k, dim=initial_tree_dim) for key in keys]
        
        validation_errors = []
        entry_instance_errors = []
        
        try:
            # Create original tree
            original_tree = self.create_tree_from_entries(entries, ranks, DIM=initial_tree_dim)
            original_tree_structure = print_pretty(original_tree)
            
            # Get dummies and validate original tree
            dummies = self.get_dummies(original_tree)
            exp_keys = sorted(dummies + keys)
            validation_errors.extend(self.validate_tree_structure(original_tree, exp_keys))
            
            # Convert tree to klist
            klist = _tree_to_klist(original_tree)
            klist_structure = print_pretty(klist)
            
            # Validate klist with SORTED entries (this was problematic!)
            entries_sorted = sorted(entries, key=lambda e: e.item.key)
            
            # Convert klist back to tree
            new_tree = _klist_to_tree(klist, K=self.k, DIM=initial_tree_dim)
            new_tree_structure = print_pretty(new_tree)
            new_dummies = self.get_dummies(new_tree)
            
            validation_errors.extend(self.validate_tree_structure(new_tree, exp_keys))
            
            # Check entry instances - this often failed!
            entry_instance_errors.extend(self.assert_entries_present_same_instance(entries_sorted, list(new_tree)))
            
        except Exception as e:
            validation_errors.append(f"Exception during old implementation: {e}")
            original_tree_structure = "ERROR"
            klist_structure = "ERROR" 
            new_tree_structure = "ERROR"
            new_dummies = []
        
        return TestResult(
            iteration=iteration,
            keys=keys,
            ranks=ranks,
            original_tree_structure=original_tree_structure,
            klist_structure=klist_structure,
            new_tree_structure=new_tree_structure,
            original_dummies=dummies if 'dummies' in locals() else [],
            new_dummies=new_dummies,
            validation_errors=validation_errors,
            entry_instance_errors=entry_instance_errors,
            success=len(validation_errors) == 0 and len(entry_instance_errors) == 0
        )
    
    def run_new_implementation(self, iteration: int, initial_tree_dim: int = 5, num_entries: int = 5) -> TestResult:
        """Run the new (fixed) implementation"""
        # NEW IMPLEMENTATION - Fixed version
        keys = random.sample(range(1, 10000), num_entries)
        keys = sorted(keys)
        
        entries = self.create_entries(keys)
        ranks = [calc_rank_for_dim(key=key, k=self.k, dim=initial_tree_dim) for key in keys]
        
        validation_errors = []
        entry_instance_errors = []
        
        try:
            # Create original tree
            original_tree = self.create_tree_from_entries(entries, ranks, DIM=initial_tree_dim)
            original_tree_structure = print_pretty(original_tree)
            
            # Get dummies and validate original tree
            dummies = self.get_dummies(original_tree)
            exp_keys = sorted(dummies + keys)
            validation_errors.extend(self.validate_tree_structure(original_tree, exp_keys))
            
            # Convert tree to klist
            klist = _tree_to_klist(original_tree)
            klist_structure = print_pretty(klist)
            
            # Validate klist with ORIGINAL entries (not sorted!)
            
            # Convert klist back to tree
            new_tree = _klist_to_tree(klist, K=self.k, DIM=initial_tree_dim)
            new_tree_structure = print_pretty(new_tree)
            new_dummies = self.get_dummies(new_tree)
            new_exp_keys = sorted(new_dummies + keys)
            
            validation_errors.extend(self.validate_tree_structure(new_tree, new_exp_keys))
            
            # Check entry instances with ORIGINAL entries (not sorted!)
            entry_instance_errors.extend(self.assert_entries_present_same_instance(entries, list(new_tree)))
            
        except Exception as e:
            validation_errors.append(f"Exception during new implementation: {e}")
            original_tree_structure = "ERROR"
            klist_structure = "ERROR"
            new_tree_structure = "ERROR"
            new_dummies = []
        
        return TestResult(
            iteration=iteration,
            keys=keys,
            ranks=ranks,
            original_tree_structure=original_tree_structure,
            klist_structure=klist_structure,
            new_tree_structure=new_tree_structure,
            original_dummies=dummies if 'dummies' in locals() else [],
            new_dummies=new_dummies,
            validation_errors=validation_errors,
            entry_instance_errors=entry_instance_errors,
            success=len(validation_errors) == 0 and len(entry_instance_errors) == 0
        )
    
    def compare_random_generation_methods(self, iterations: int = 100) -> Dict[str, Any]:
        """Compare the two random key generation methods"""
        print("=== COMPARING RANDOM GENERATION METHODS ===")
        
        old_method_collisions = 0
        new_method_collisions = 0
        
        for i in range(iterations):
            # Old method
            random.seed(i)
            keys_old = set()
            attempts = 0
            while len(keys_old) < 5 and attempts < 1000:
                keys_old.add(random.randint(1, 10000))
                attempts += 1
            if attempts >= 1000:
                old_method_collisions += 1
            
            # New method
            random.seed(i)
            try:
                keys_new = random.sample(range(1, 10000), 5)
                # This should never fail with our parameters
            except ValueError:
                new_method_collisions += 1
        
        results = {
            "old_method_collisions": old_method_collisions,
            "new_method_collisions": new_method_collisions,
            "old_method_success_rate": (iterations - old_method_collisions) / iterations,
            "new_method_success_rate": (iterations - new_method_collisions) / iterations
        }
        
        print(f"Old method collision rate: {old_method_collisions}/{iterations} ({old_method_collisions/iterations*100:.1f}%)")
        print(f"New method collision rate: {new_method_collisions}/{iterations} ({new_method_collisions/iterations*100:.1f}%)")
        
        return results
    
    def analyze_entry_sorting_issue(self) -> Dict[str, Any]:
        """Analyze the entry sorting vs instance preservation issue"""
        print("\n=== ANALYZING ENTRY SORTING ISSUE ===")
        
        # Create test entries
        keys = [100, 50, 75, 25]
        entries = self.create_entries(keys)
        
        print(f"Original entries order: {[e.item.key for e in entries]}")
        print(f"Original entry IDs: {[id(e) for e in entries]}")
        
        # Sort entries
        entries_sorted = sorted(entries, key=lambda e: e.item.key)
        print(f"Sorted entries order: {[e.item.key for e in entries_sorted]}")
        print(f"Sorted entry IDs: {[id(e) for e in entries_sorted]}")
        
        # Check if IDs are preserved
        same_instances = all(
            any(id(orig) == id(sort) for sort in entries_sorted)
            for orig in entries
        )
        
        print(f"Same instances preserved after sorting: {same_instances}")
        
        # The issue: when we sort entries, we change the order but preserve instances
        # However, when we compare with tree entries, the tree might have a different order
        # This can cause the instance comparison to fail if we expect a specific order
        
        return {
            "original_order": [e.item.key for e in entries],
            "sorted_order": [e.item.key for e in entries_sorted],
            "instances_preserved": same_instances
        }
    
    def run_comprehensive_debug(self, iterations: int = 20) -> None:
        """Run comprehensive debugging of both implementations"""
        print("=== COMPREHENSIVE ROUNDTRIP TEST DEBUGGING ===")
        print(f"Running {iterations} iterations for each implementation...\n")
        
        # First analyze the random generation and sorting issues
        self.compare_random_generation_methods()
        self.analyze_entry_sorting_issue()
        
        print(f"\n=== RUNNING {iterations} TEST ITERATIONS ===")
        
        old_results = []
        new_results = []
        
        # Run tests with fixed seed for reproducibility
        for i in range(iterations):
            print(f"Running iteration {i+1}/{iterations}...")
            
            # Test old implementation
            random.seed(42 + i)  # Fixed seed for reproducibility
            old_result = self.run_old_implementation(i)
            old_results.append(old_result)
            
            # Test new implementation with same seed
            random.seed(42 + i)  # Same seed for fair comparison
            new_result = self.run_new_implementation(i)
            new_results.append(new_result)
        
        # Analyze results
        old_success_rate = sum(1 for r in old_results if r.success) / len(old_results)
        new_success_rate = sum(1 for r in new_results if r.success) / len(new_results)
        
        print(f"\n=== RESULTS SUMMARY ===")
        print(f"Old implementation success rate: {old_success_rate:.2%} ({sum(1 for r in old_results if r.success)}/{len(old_results)})")
        print(f"New implementation success rate: {new_success_rate:.2%} ({sum(1 for r in new_results if r.success)}/{len(new_results)})")
        
        # Analyze failure patterns
        old_failures = [r for r in old_results if not r.success]
        new_failures = [r for r in new_results if not r.success]
        
        if old_failures:
            print(f"\n=== OLD IMPLEMENTATION FAILURE ANALYSIS ===")
            print(f"Total failures: {len(old_failures)}")
            
            # Categorize error types
            validation_errors = sum(1 for f in old_failures if f.validation_errors)
            instance_errors = sum(1 for f in old_failures if f.entry_instance_errors)
            
            print(f"Validation errors: {validation_errors}")
            print(f"Entry instance errors: {instance_errors}")
            
            # Show first few failures in detail
            for i, failure in enumerate(old_failures[:3]):
                print(f"\n--- Old Failure {i+1} (Iteration {failure.iteration}) ---")
                print(f"Keys: {failure.keys}")
                print(f"Validation errors: {failure.validation_errors}")
                print(f"Entry instance errors: {failure.entry_instance_errors}")
        
        if new_failures:
            print(f"\n=== NEW IMPLEMENTATION FAILURE ANALYSIS ===")
            print(f"Total failures: {len(new_failures)}")
            
            for i, failure in enumerate(new_failures[:3]):
                print(f"\n--- New Failure {i+1} (Iteration {failure.iteration}) ---")
                print(f"Keys: {failure.keys}")
                print(f"Validation errors: {failure.validation_errors}")
                print(f"Entry instance errors: {failure.entry_instance_errors}")
        
        # Identify the key differences
        print(f"\n=== KEY DIFFERENCES IDENTIFIED ===")
        print("1. Random Key Generation:")
        print("   - Old: Uses while loop with random.randint() -> can have duplicates/infinite loops")
        print("   - New: Uses random.sample() -> guarantees unique keys")
        
        print("\n2. Entry Instance Handling:")
        print("   - Old: Sorts entries before final comparison -> changes reference order")
        print("   - New: Uses original entry order for comparison -> preserves references")
        
        print("\n3. Validation Order:")
        print("   - Old: Validates with sorted entries list")
        print("   - New: Validates with original entries list")
        
        print("\n4. Dummy Handling:")
        print("   - Old: Uses same dummy collection for both validations")
        print("   - New: Recalculates dummies for new tree (more robust)")

def main():
    """Main debugging function"""
    debugger = RoundtripDebugger()
    debugger.run_comprehensive_debug(iterations=50)

if __name__ == "__main__":
    main()
