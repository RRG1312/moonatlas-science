# Releasing science

A **merge** changes this repository. A **release** publishes a versioned artifact. Applications — the MOONATLAS
application included — change only when *they* choose to pin a new release. Nothing in this repository deploys,
pushes to or modifies any application.

```
pull request → CI → human review → merge                (this repository changes)
  → maintainer cuts a release                           (a new versioned artifact exists)
  → an application pins that release by name + SHA-256 (the application changes, on its own schedule)
```

## When a release is needed

| Change | Release? |
|--------|----------|
| Documentation, tests, refactors with byte-identical output | No |
| Any published value, field, identifier rule or generated text | Yes, with a new `processingVersion` (`docs/reproducibility/versioning.md`) |
| New dataset, task or model checkpoint | Yes (minor) |

## Cutting a release

On the machine that holds the inference artifacts (`MOONATLAS_DATA_ROOT`), with `processingVersion` bumped in
`src/moonatlas_science/config.py`:

1. Normalize every task from its raw artifacts, then `mosaic`, `catalog` and `manifest` into a new build directory
   (`MOONATLAS_BUILD_DIR`). Re-run inference only when the model outputs themselves must change.
2. `moonatlas-science validate --root <build>` — full scope must pass.
3. `moonatlas-science publish` and `moonatlas-science validate --root <published>` — full scope must pass.
4. When values are expected to stay the same, compare the new published dataset with the previous release at the
   value level and record the result in `CHANGELOG.md`.
5. `moonatlas-science package --out dist` — writes the archive and its manifest (SHA-256, sizes, file count,
   dataset and processing manifest checksums).
6. Regenerate `examples/science-smoke` from the smoke build of the same version, then `schemas/` and
   `examples/conformance/`, and commit them with the version bump.
7. Publish the archive and its manifest as a GitHub Release named after the `processingVersion`, with `NOTICE`
   and the upstream citations in the release notes.

The release checklist in `docs/architecture/public-artifact.md` applies before step 7.
