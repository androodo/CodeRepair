# Test image

Build the fixed local runner before a test-enabled repair or evaluation:

```bash
docker build -f docker/test-runner.Dockerfile -t coderepair-sandbox:py311 .
```

The build may use the network to fetch the pinned `pytest` wheel. Actual test
containers run with `--network none`, a read-only root filesystem, a read-only
repository mount, a non-root UID, dropped capabilities, and resource limits.
CodeRepair never installs a target repository's dependencies automatically. A
team that needs more approved dependencies should review and build a separate
image, then select its local tag with `--test-image`.

Container isolation reduces risk; it is not equivalent to a hardened virtual
machine and is not a perfect security boundary.
