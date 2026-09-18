# Publishing a dataset

A *build* is what the pipeline writes under `data/build/<processingVersion>/`. A *dataset* is a published copy
of a validated build. An *artifact* is a single reproducible archive of a dataset.

```
build (local)  →  moonatlas-science validate  →  publish  →  package  →  artifact + manifest
```

## publish

`moonatlas-science publish` (`steps/build_dataset.py`) refuses to run on an invalid build. It then applies the
publication policy, which is packaging only — no value is recomputed:

- per-patch ICE **display** rasters are dropped because they are pixel-identical to the corresponding window of
  the pole mosaic; the script re-verifies that equivalence before dropping them and sets `displayPath` to null;
- per-tile crater detail JSON is written compactly, asserting it decodes to the same object;
- everything else is copied unchanged.

The manifest is then rebuilt for the published set and the dataset is validated again.

## package

`moonatlas-science package` writes `<name>.tar.gz` plus a JSON manifest carrying `sha256`, sizes, file count and
the dataset's own manifest checksums. The archive is byte-reproducible from the same dataset: entries sorted by
path, fixed mtime/uid/gid/mode, USTAR format, gzip without a timestamp. `tests/test_packaging.py` asserts it.

Consumers are expected to verify the sha256 before using an artifact.

## What a dataset is not

A published dataset does **not** contain the raw inference artifacts: those stay on the machine that ran the
inference. Validate a published dataset with `--scope published`, which runs every check except the
cross-checks against local raw artifacts.
