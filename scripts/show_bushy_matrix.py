"""Print the bushy pile-tree attack as a rank-dimension matrix.

Rows = dimensions, columns = keys (ascending <=_U order), each cell = that
key's rank in that dimension. Ranks != 1 are highlighted red, exactly like
scripts/analyze_adversarial_keys_scenarios.py (lines 227-233). Uses the real
generator adversarial.mixed_rank._bushy_leaf_profiles, so the pattern is
authentic, not hand-built.

Examples
--------
    python scripts/show_bushy_matrix.py                     # B=2, d=3, leaf=1
    python scripts/show_bushy_matrix.py -b 2 -d 4 -l 1
    python scripts/show_bushy_matrix.py -b 3 -d 2 -l 2
    python scripts/show_bushy_matrix.py -b 2 -d 3 --profiles # also list per-key profiles
    python scripts/show_bushy_matrix.py --no-color | cat      # plain, pipe-friendly
"""

import argparse
import sys
from pathlib import Path

# Make `adversarial` importable no matter how / from where this is run.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adversarial.mixed_rank import _bushy_leaf_profiles  # the authentic generator


def _closed_form_height(branching: int, depth: int) -> int:
    # H_d = (B^{d+1} - B) / (B - 1)  (nesting-aware height, L16)
    return (branching ** (depth + 1) - branching) // (branching - 1)


def print_matrix(depth: int, branching: int, leaf_size: int, *, color: bool, profiles: bool) -> None:
    profs = _bushy_leaf_profiles(depth, branching, leaf_size)  # per-key, ascending <=_U order
    n = len(profs)
    rank_lists = [[p[j] for p in profs] for j in range(depth)]  # transpose -> dimension-major

    print(
        f"Bushy pile-tree rank matrix: B={branching}, d={depth}, leaf={leaf_size}, "
        f"n = B^d*leaf = {n} keys, "
        f"predicted height H_d = {_closed_form_height(branching, depth)}"
    )
    for j, rl in enumerate(rank_lists, start=1):
        display_rl = rl
        if len(rl) > 20:  # same truncation the source script uses
            display_rl = [*rl[:10], "...", *rl[-10:]]
        cells = []
        for num in display_rl:
            if color and num != 1 and num != "...":
                cells.append(f"\033[31m{num}\033[0m")
            else:
                cells.append(str(num))
        print(f"dim {j}:  " + " ".join(cells))

    if profiles:
        print("\nper-key profiles (column k = (rho_1, ..., rho_d)):")
        for k, p in enumerate(profs):
            print(f"  k{k}: {tuple(p)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--depth", type=int, default=3, help="attack depth = number of dimensions (default 3)")
    ap.add_argument("-b", "--branching", type=int, default=2, help="branching B = piles per dimension (default 2)")
    ap.add_argument("-l", "--leaf", type=int, default=1, help="leaf size = keys per leaf-path (default 1)")
    ap.add_argument("--profiles", action="store_true", help="also list each key's full profile")
    ap.add_argument("--no-color", dest="color", action="store_false", help="disable red highlighting (for piping)")
    args = ap.parse_args()

    if args.branching < 2:
        ap.error("branching must be >= 2 (a bushy path needs >= 2 nested piles)")
    if args.depth < 1 or args.leaf < 1:
        ap.error("depth and leaf must be >= 1")

    print_matrix(args.depth, args.branching, args.leaf, color=args.color, profiles=args.profiles)


if __name__ == "__main__":
    main()
