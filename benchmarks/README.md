# Development benchmark fixtures

`tasks.json` contains three small, deliberately broken Python projects. The
source directories contain visible tests. Independent target and regression
tests live under `hidden/` and are mounted only after an agent run ends.

Each plain source tree is pinned by CodeRepair's canonical v1 SHA-256. If a
fixture is intentionally changed, recompute its hash explicitly:

```bash
python -c "from coderepair.evaluation import canonical_source_tree_sha256 as h; print(h('benchmarks/fixtures/chunking'))"
```

Then review both the fixture change and manifest pin. These tasks exercise the
evaluation machinery; their outcomes must not be presented as general coding
agent performance.
