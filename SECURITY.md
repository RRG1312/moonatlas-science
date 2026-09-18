# Security Policy

MOONATLAS Science is an offline data-processing pipeline. It has no server, no user accounts, no database and
no runtime service. The realistic risk surface is therefore: code that executes on a contributor's machine,
supply-chain integrity of the upstream models and datasets, and integrity of the published scientific outputs.

## Supported versions

Security fixes land on `main` and, when relevant, in the next `science-<version>` build. Older builds are not
patched; rebuild from `main` instead.

## Reporting a vulnerability

Please use **GitHub Security Advisories** ("Report a vulnerability" on the Security tab) so the report stays
private until a fix exists. If that is unavailable to you, open a normal issue that says only *"security report,
please provide a private channel"* — do not include details.

Please include: what an attacker can do, the minimal reproduction, affected files or commands, and the
environment. **Do not include credentials, tokens or private paths.**

Expect an acknowledgement within a few days. Disclosure is coordinated: we agree on a timeline and credit you
in the release notes unless you prefer otherwise.

## In scope

- Code execution triggered by processing untrusted input (a crafted GeoTIFF, JSON, checkpoint or archive).
- Path traversal or arbitrary file write from dataset, archive or configuration handling.
- Checksum/verification bypass in the download and packaging scripts (supply-chain integrity).
- Dependency vulnerabilities that this project's usage actually exposes.
- A way to make the pipeline publish a value that silently contradicts its provenance.

## Out of scope

- Vulnerabilities in upstream models, datasets or libraries — report those upstream; tell us if our pinned
  revision is affected so we can move the pin.
- Running untrusted checkpoints or datasets on purpose: loading a model executes code by design. Only fetch
  from the pinned official sources.
- The MOONATLAS application, which is a separate product: report issues with it through the contact listed on its site.
- Results you disagree with scientifically. That is a
  [scientific/data issue](.github/ISSUE_TEMPLATE/scientific_data_issue.yml), not a vulnerability.

## Hardening expectations for contributors

- Never commit credentials, tokens, `.env` files, checkpoints or downloaded datasets. Tokens for upstream
  registries belong in your environment, never in the repository.
- Verify checksums for anything downloaded; a new source without a pinned revision and checksum will not be
  merged.
- Avoid `pickle`/`torch.load` on untrusted paths; keep strict state-dict loading.
- Treat archive extraction as hostile input: validate member paths before writing.
