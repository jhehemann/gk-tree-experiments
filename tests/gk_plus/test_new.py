def test_roundtrip_tree_to_klist_to_tree(self):
        """Test the round trip from tree to KList and back to tree"""
        iterations = 1000
        initial_tree_dim = 1  # Start with dimension 1 for simpler debugging
        self.k = 4
        num_entries = 5
        
        # Use a fixed seed for reproducibility
        random.seed(42)
        
        for iteration in range(iterations):
            with self.subTest(iteration=iteration):
                # Generate unique keys deterministically 
                keys = random.sample(range(1, 10000), num_entries)
                keys = sorted(keys)
                entries = self.create_entries(keys)
                ranks = [calc_rank_for_dim(key=key, k=self.k, dim=initial_tree_dim) for key in keys]
                
                msg = f"\n\nIteration {iteration}: dim = {initial_tree_dim}"
                msg += f"\nkeys = {keys}"
                msg += f"\nranks = {ranks}"
                
                # Log rank information for debugging
                if hasattr(self, '_log_ranks'):
                    self._log_ranks(self.k, keys, 10)
                    self._log_ranks(self.k, list(range(-11, 0, -1)))
                
                # Create original tree
                original_tree = self.create_tree_from_entries(entries, ranks, DIM=initial_tree_dim)
                msg += f"\noriginal_tree = {print_pretty(original_tree)}"
                
                # Get dummies and validate original tree
                dummies = self.get_dummies(original_tree)
                exp_keys = sorted(dummies + keys)
                self.validate_tree(original_tree, exp_keys, msg)
                
                # Convert tree to klist
                klist = _tree_to_klist(original_tree)
                msg += f"\nklist = {print_pretty(klist)}"
                
                # Validate klist (entries should match original entries, not including dummies)
                self.validate_klist(klist, entries, msg)
                
                # Convert klist back to tree
                new_tree = _klist_to_tree(klist, K=self.k, DIM=initial_tree_dim)
                msg += f"\nnew_tree = {print_pretty(new_tree)}"
                
                # Validate new tree structure and keys
                new_dummies = self.get_dummies(new_tree)
                new_exp_keys = sorted(new_dummies + keys)
                self.validate_tree(new_tree, new_exp_keys, msg)
                
                # Check that original entries are present in new tree
                # Note: We compare the original entries (not sorted) with the new tree
                self.assert_entries_present_same_instance(entries, list(new_tree))


def test_roundtrip_tree_to_klist_to_tree(self):
        """Test the round trip from tree to KList and back to tree"""
        # Create a random tree
        iterations = 1000
        initial_tree_dim = 5
        self.k = 4
        num_entries = 5
        for _ in range(iterations):
            with self.subTest(iteration=_):
                keys = set()
                while len(keys) < num_entries:
                    keys.add(random.randint(1, 10000))
                keys = sorted(list(keys))
                entries = self.create_entries(keys)
                ranks = [calc_rank_for_dim(key=key, k=self.k, dim=initial_tree_dim) for key in keys]
                
                msg = f"\n\n\nIteration {_}: dim = {initial_tree_dim}"
                msg += f"\n\nkeys = {keys}"
                msg += f"\nranks = {ranks}"
                
                self._log_ranks(self.k, keys, 10)
                self._log_ranks(self.k, list(range(-11, 0, -1)))
                original_tree = self.create_tree_from_entries(entries, ranks, DIM=initial_tree_dim)
                msg += f"\n\noriginal_tree = {print_pretty(original_tree)}"
                dummies = self.get_dummies(original_tree)
                keys_sorted = sorted(keys)
                exp_keys = sorted(dummies + keys_sorted)
                self.validate_tree(original_tree, exp_keys, msg)
                klist = _tree_to_klist(original_tree)
                msg += f"\n\nklist = {print_pretty(klist)}"
                entries_sorted = sorted(entries, key=lambda e: e.item.key)
                self.validate_klist(klist, entries_sorted, msg)
                new_tree = _klist_to_tree(klist, K=self.k, DIM=initial_tree_dim)
                msg += f"\n\nnew_tree = {print_pretty(new_tree)}"
                self.validate_tree(new_tree, exp_keys, msg)
                self.assert_entries_present_same_instance(entries_sorted, list(new_tree))