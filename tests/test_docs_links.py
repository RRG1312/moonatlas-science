"""Every relative link in the documentation points at a file that exists."""

from __future__ import annotations

import re

import pytest

from moonatlas_science import config

ROOT = config.PROJECT_ROOT
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
DOCUMENTS = sorted(
    p for p in ROOT.rglob("*.md")
    if not any(part.startswith(".") and part not in (".github",) for part in p.relative_to(ROOT).parts)
)


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_relative_links_resolve(document):
    text = document.read_text(encoding="utf-8")
    for target in LINK.findall(text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = (document.parent / target.split("#", 1)[0]).resolve()
        assert path.exists(), f"{document.relative_to(ROOT)} → {target}"
