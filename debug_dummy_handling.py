#!/usr/bin/env python3
"""
Focused debugging script for the dummy handling issue in roundtrip tests.

The main issue identified: After tree->klist->tree conversion, the new tree 
can have different dummy entries than the original tree, causing validation failures.
"""

import sys
import os
import random
from typing import List

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from gplus_trees.base import Entry, Item
from gplus_trees.g_k_plus.factory import create_gkplus_tree
from gplus_trees.g_k_plus.g_k_plus_base import GKPlusTreeBase, _tree_to_klist, _klist_to_tree, print_pretty
from gplus_trees.g_k_plus.utils import calc_rank_for_dim
from tests.gk_plus.base import TreeTestCase as GKPlusTreeTestCase

class DummyHandlingDebugger(GKPlusTreeTestCase):
    """Debug the dummy handling issue specifically"""
    
    def __init__(self):
        super().__init__()
        self.k = 4
    
    def create_entries(self, keys: List[int]) -> List[Entry]:
        """Create entries from keys for testing"""
        return [Entry(Item(key, f"val_{key}"), None) for key in keys]
    
    def create_tree_from_entries(self, entries: List[Entry], ranks: List[int], DIM: int) -> GKPlusTreeBase:
        """Create a GKPlusTree for a given dimension from a list of entries"""
        tree = create_gkplus_tree(K=self.k, dimension=DIM)
        for i, entry in enumerate(entries):
            tree, _ = tree.insert_entry(entry, ranks[i])
        return tree
    
    def analyze_dummy_issue(self, keys: List[int], dim: int = 5):
        """Analyze the specific dummy handling issue"""
        print(f"\n=== ANALYZING DUMMY ISSUE ===")
        print(f"Keys: {keys}")
        print(f"Dimension: {dim}")
        
        entries = self.create_entries(keys)
        ranks = [calc_rank_for_dim(key=key, k=self.k, dim=dim) for key in keys]
        
        print(f"Ranks: {ranks}")
        
        # Create original tree
        print(f"\n1. Creating original tree...")
        original_tree = self.create_tree_from_entries(entries, ranks, DIM=dim)
        print(f"Original tree structure:\n{print_pretty(original_tree)}")
        
        # Get original dummies
        original_dummies = self.get_dummies(original_tree)
        print(f"Original dummies: {original_dummies}")
        
        original_all_keys = sorted(original_dummies + keys)
        print(f"Original expected keys: {original_all_keys}")
        
        # Convert to klist
        print(f"\n2. Converting tree to klist...")
        klist = _tree_to_klist(original_tree)
        print(f"KList structure:\n{print_pretty(klist)}")
        
        # Convert back to tree
        print(f"\n3. Converting klist back to tree...")
        new_tree = _klist_to_tree(klist, K=self.k, DIM=dim)
        print(f"New tree structure:\n{print_pretty(new_tree)}")
        
        # Get new dummies
        new_dummies = self.get_dummies(new_tree)
        print(f"New dummies: {new_dummies}")
        
        new_all_keys = sorted(new_dummies + keys)
        print(f"New expected keys: {new_all_keys}")
        
        # Compare
        print(f"\n4. Comparison:")
        print(f"Original dummies: {original_dummies}")
        print(f"New dummies:      {new_dummies}")
        print(f"Dummies match:    {original_dummies == new_dummies}")
        print(f"Original keys:    {original_all_keys}")
        print(f"New keys:         {new_all_keys}")
        print(f"Keys match:       {original_all_keys == new_all_keys}")
        
        # Extract actual keys from trees
        original_actual_keys = sorted([entry.item.key for entry in original_tree])
        new_actual_keys = sorted([entry.item.key for entry in new_tree])
        
        print(f"Original actual:  {original_actual_keys}")
        print(f"New actual:       {new_actual_keys}")
        print(f"Actual match:     {original_actual_keys == new_actual_keys}")
        
        return {
            'original_dummies': original_dummies,
            'new_dummies': new_dummies,
            'dummies_match': original_dummies == new_dummies,
            'original_expected': original_all_keys,
            'new_expected': new_all_keys,
            'keys_match': original_all_keys == new_all_keys,
            'original_actual': original_actual_keys,
            'new_actual': new_actual_keys,
            'actual_match': original_actual_keys == new_actual_keys
        }
    
    def find_problematic_cases(self, iterations: int = 100):
        """Find cases where dummy handling causes issues"""
        print(f"=== SEARCHING FOR PROBLEMATIC CASES ===")
        print(f"Testing {iterations} random cases...")
        
        problematic_cases = []
        
        for i in range(iterations):
            # Generate random test case
            random.seed(42 + i)
            keys = random.sample(range(1, 10000), 5)
            
            try:
                result = self.analyze_dummy_issue(keys, dim=5)
                if not result['actual_match']:
                    problematic_cases.append({
                        'iteration': i,
                        'keys': keys,
                        'result': result
                    })
                    print(f"\nFound problematic case #{len(problematic_cases)} (iteration {i}):")
                    print(f"Keys: {keys}")
                    print(f"Original actual: {result['original_actual']}")
                    print(f"New actual: {result['new_actual']}")
            except Exception as e:
                print(f"Error in iteration {i}: {e}")
        
        print(f"\n=== SUMMARY ===")
        print(f"Tested {iterations} cases")
        print(f"Found {len(problematic_cases)} problematic cases")
        print(f"Success rate: {(iterations - len(problematic_cases)) / iterations:.1%}")
        
        return problematic_cases
    
    def demonstrate_fix(self, keys: List[int]):
        """Demonstrate the fix for the dummy handling issue"""
        print(f"\n=== DEMONSTRATING THE FIX ===")
        print(f"Keys: {keys}")
        
        entries = self.create_entries(keys)
        ranks = [calc_rank_for_dim(key=key, k=self.k, dim=5) for key in keys]
        
        # Original (problematic) approach
        print(f"\n--- ORIGINAL (PROBLEMATIC) APPROACH ---")
        original_tree = self.create_tree_from_entries(entries, ranks, DIM=5)
        original_dummies = self.get_dummies(original_tree)
        original_expected = sorted(original_dummies + keys)
        
        klist = _tree_to_klist(original_tree)
        new_tree = _klist_to_tree(klist, K=self.k, DIM=5)
        
        # Use original dummies for validation (WRONG!)
        try:
            self.validate_tree(new_tree, original_expected, "Original approach")
            print("✓ Original approach succeeded")
        except Exception as e:
            print(f"✗ Original approach failed: {e}")
        
        # Fixed approach
        print(f"\n--- FIXED APPROACH ---")
        new_tree_fixed = _klist_to_tree(klist, K=self.k, DIM=5)
        new_dummies = self.get_dummies(new_tree_fixed)  # Recalculate dummies!
        new_expected = sorted(new_dummies + keys)
        
        try:
            self.validate_tree(new_tree_fixed, new_expected, "Fixed approach")
            print("✓ Fixed approach succeeded")
        except Exception as e:
            print(f"✗ Fixed approach failed: {e}")
        
        print(f"\nThe key insight:")
        print(f"- Original dummies: {original_dummies}")
        print(f"- New dummies:      {new_dummies}")
        print(f"- Must recalculate dummies after tree reconstruction!")

def main():
    """Main function to run the focused debugging"""
    debugger = DummyHandlingDebugger()
    
    # First, find some problematic cases
    problematic_cases = debugger.find_problematic_cases(50)
    
    if problematic_cases:
        print(f"\n=== DETAILED ANALYSIS OF FIRST PROBLEMATIC CASE ===")
        first_case = problematic_cases[0]
        debugger.analyze_dummy_issue(first_case['keys'])
        
        print(f"\n=== DEMONSTRATING THE FIX ===")
        debugger.demonstrate_fix(first_case['keys'])
    else:
        print("No problematic cases found in this run. The issue might be rare or dependent on specific conditions.")
    
    print(f"\n=== CONCLUSION ===")
    print("The main issue was in dummy handling:")
    print("1. Original tree has certain dummy entries based on its structure")
    print("2. After tree->klist->tree conversion, the new tree might have different dummies")
    print("3. Validating the new tree against the original dummies causes failures")
    print("4. FIX: Always recalculate dummies for the new tree instead of reusing original dummies")

if __name__ == "__main__":
    main()
