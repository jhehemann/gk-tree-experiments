# Experiment journal

Durable findings from experiment campaigns — the conclusions that otherwise live only in
commit messages or discarded logs.

Run logs under `stats/logs/` are ephemeral (gitignored, no retention) and benchmark
artifacts live outside the repo (`../.isolated-benchmarks/`, by design). A conclusion
worth keeping belongs **here**, not in the logs.

## Format

One file per entry: `YYYY-MM-DD-<slug>.md`, containing:

- **Date / Context** — which campaign, parameters (counts, K, dim, flags)
- **Observation** — what the runs showed
- **Conclusion** — what follows from it
- **References** — commit hashes, fixture files, `../.isolated-benchmarks/` run dirs

No speculative entries: write one only when there is a real finding.
