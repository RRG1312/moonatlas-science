# Determinism

Same inputs and same pinned revisions must produce byte-identical outputs. What that requires:

- **Ordering.** Every iteration that reaches an output is sorted by a stable key (sample id, path, identity
  key). Set and dict iteration never leaks into published order.
- **Numbers.** Floats are written through the project's formatting helpers; no locale-dependent formatting.
- **Timestamps.** `generatedAt` comes from the build inputs, not from the wall clock, and archive entries use a
  fixed mtime.
- **Identity.** Public discovery ids come from the append-only registry, so a rebuild cannot renumber anything.
- **Checksums.** `processing-manifest.json` records sha256 and size for every published file, and
  `moonatlas-science package` produces a byte-reproducible archive (sorted entries, fixed mtime/uid/gid/mode,
  USTAR, gzip without timestamp).

Verifying a rebuild:

```bash
moonatlas-science package --out dist            # before
# rebuild, then package again into another directory
sha256sum dist/*.tar.gz other/*.tar.gz         # the digests must match
```

A change that makes output non-deterministic is rejected even when the science is right, because it removes the
ability to check the science.
