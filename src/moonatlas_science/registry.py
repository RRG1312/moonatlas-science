"""Append-only discovery registry: public discovery ids that survive rebuilds (docs/provenance/data-contract.md › Discovery System).

An entry binds one scientific identity key to one public id forever. Entries are never edited or removed, ids are
never reused, and new ids are allocated in identity-key order, so neither upstream ordering nor array positions can
change them. Anything inconsistent raises RegistryError instead of guessing.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import config

FORMAT = "moonatlas-discovery-registry-v1"
PREFIX = {"CRATER_CANDIDATE": "CR", "HIGH_PROSPECTIVITY_REGION": "ICE", "IMP_CANDIDATE": "IMP"}
ID_PATTERN = re.compile(r"^MM-(CR|ICE|IMP)-(\d{6})$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Committed, append-only: public ids must survive a rebuild. Override only for an experiment.
REGISTRY_PATH = Path(os.environ.get("MOONATLAS_REGISTRY", config.PROJECT_ROOT / "configs" / "discovery-registry.json"))


class RegistryError(ValueError):
    """The registry is corrupt, or an identity conflicts with what it already records."""


class DiscoveryRegistry:
    def __init__(self, entries: list[dict] | None = None, number_base: int = 0):
        # number_base only offsets synthetic fixture ids (MM-*-9xxxxx) so they can never look like real ones.
        self.number_base = number_base
        self.entries: list[dict] = []
        self._by_key: dict[str, dict] = {}
        self._by_id: dict[str, dict] = {}
        for entry in entries or []:
            self._add(dict(entry))

    def _add(self, entry: dict) -> None:
        match = ID_PATTERN.match(str(entry.get("id", "")))
        if (set(entry) != {"id", "type", "key", "firstCatalogued"} or not match
                or PREFIX.get(entry["type"]) != match.group(1) or not isinstance(entry["key"], str) or not entry["key"]
                or not DATE_PATTERN.match(str(entry["firstCatalogued"]))):
            raise RegistryError(f"malformed registry entry: {entry}")
        if entry["id"] in self._by_id:
            raise RegistryError(f"discovery id {entry['id']} is registered twice")
        if entry["key"] in self._by_key:
            raise RegistryError(f"identity {entry['key']!r} is registered as both {self._by_key[entry['key']]['id']} "
                                f"and {entry['id']}")
        self.entries.append(entry)
        self._by_key[entry["key"]] = entry
        self._by_id[entry["id"]] = entry

    @classmethod
    def load(cls, path: Path = REGISTRY_PATH) -> "DiscoveryRegistry":
        if not path.exists():
            return cls()
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RegistryError(f"{path}: not valid JSON ({error})") from error
        if not isinstance(document, dict) or document.get("format") != FORMAT or not isinstance(document.get("entries"), list):
            raise RegistryError(f"{path}: expected {{'format': {FORMAT!r}, 'entries': [...]}}")
        return cls(document["entries"])

    def save(self, path: Path = REGISTRY_PATH) -> None:
        # Append-only: everything already on disk must still be here, unchanged.
        previous = DiscoveryRegistry.load(path)
        for entry in previous.entries:
            if self._by_id.get(entry["id"]) != entry:
                raise RegistryError(f"refusing to save: registered entry {entry['id']} would change or disappear")
        entries = sorted(self.entries, key=lambda e: (e["type"], e["id"]))
        text = json.dumps({"format": FORMAT, "entries": entries}, indent=2, ensure_ascii=False) + "\n"
        path.write_text(text, encoding="utf-8", newline="\n")

    def entries_with_prefix(self, key_prefix: str) -> list[dict]:
        return [e for e in self.entries if e["key"].startswith(key_prefix)]

    def assign(self, requests: dict[str, str], date: str) -> dict[str, dict]:
        """Map identity key → registry entry for {key: discovery type}, registering unknown keys in key order."""
        for key, dtype in requests.items():
            if dtype not in PREFIX:
                raise RegistryError(f"unknown discovery type {dtype!r}")
            known = self._by_key.get(key)
            if known and known["type"] != dtype:
                raise RegistryError(f"identity {key!r} is registered as {known['type']}, requested as {dtype}")
        for key in sorted(k for k in requests if k not in self._by_key):
            prefix = PREFIX[requests[key]]
            used = [int(ID_PATTERN.match(e["id"]).group(2)) for e in self.entries if e["id"].startswith(f"MM-{prefix}-")]
            number = max(used, default=self.number_base) + 1
            if number > 999_999:
                raise RegistryError(f"discovery id space for MM-{prefix} is exhausted")
            self._add({"id": f"MM-{prefix}-{number:06d}", "type": requests[key], "key": key, "firstCatalogued": date})
        return {key: self._by_key[key] for key in requests}
