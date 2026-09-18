# Public science artifact

A published science build can be distributed as a single archive (`moonatlas-science package`). This page states
what such an archive contains, what it must not contain, and how it relates to applications that consume it.

## What a published build contains

Everything in it is produced by the public pipeline in this repository and is documented by the data contract
(`docs/provenance/data-contract.md`):

| Content | Class |
|---------|-------|
| Crater regions, detections and per-tile detail | Normalized model output + SomBench-derived reference boxes |
| Ice sectors and per-pole mosaics, raw value rasters | Normalized model output + SomBench-derived reference rasters |
| IMP regions, masks and observation groups | Normalized model output + SomBench-derived reference masks |
| Previews and display rasters | Display representations of SomBench-derived imagery and of model outputs |
| Discoveries and global statistics | The curated catalogue produced by the public selection rules and the committed registry |
| `sources.json`, `manifest.json`, `processing-manifest.json` | Provenance, identity and checksums |

It contains no code, no product configuration, no branding assets, no checkpoints and no original upstream files.
All third-party-derived content is covered by SomBench's CC BY 4.0 licence and must ship with `NOTICE`.

## Public science artifact vs. application data

| | Public science artifact | Application data |
|---|---|---|
| Produced by | `moonatlas-science publish` + `package` | The application's own build |
| Contents | The published build above | Whatever the application derives from it: staging, caching, loading plans, interface state |
| Where it lives | A release of this repository | The application's own repository and hosting |
| Interface | The data contract and `schemas/` | Private to the application |

An application consumes the public artifact exactly as any other consumer would: by downloading a release and
verifying its sha256. It decides when to move to a new release; nothing in this repository updates, deploys or
pushes to an application, so merging a pull request here never changes a running website. Nothing application-specific belongs in the artifact, and nothing in the artifact is needed
to reconstruct an application's interface — the interface is not data.

## Release checklist

Before attaching an archive to a public release:

1. It was produced by `moonatlas-science publish` and `package` from a build that passes
   `moonatlas-science validate` in full scope on the machine that ran the inference.
2. Its `processingVersion` is new, or its bytes are identical to the previous release of that version.
3. Its generated text (discovery descriptions, labels) matches the project's current name and wording rules — a
   text change is an output change and needs a new `processingVersion`.
4. `NOTICE` and the build's `sources.json` travel with it, and the release notes cite the upstream work.
5. The sha256 in the release notes matches the archive's manifest.
