#!/usr/bin/env python
"""Runs the code blocks of site/**/*.html, in both languages.

An example that does not run misleads the reader, so every block is executed.
Each page has its own namespace, and within a page the blocks share names, in
order. The blocks run in a temporary directory, because the ones that export
models write files, and those must not end up in the repository.

    python tools/check_site_examples.py
"""
from __future__ import annotations

import html
import os
import re
import sys
import tempfile
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
BLOCK = re.compile(r'<pre><code(?P<attr>[^>]*)>(?P<src>.*?)</code></pre>', re.S)


def blocks(page: Path):
    text = page.read_text(encoding="utf-8")
    for i, m in enumerate(BLOCK.finditer(text), start=1):
        if "plain" in m.group("attr"):
            continue
        line = text[:m.start()].count("\n") + 1
        yield i, line, html.unescape(m.group("src")).strip()


def main() -> int:
    pages = sorted(SITE.rglob("*.html"))
    failures = 0
    for page in pages:
        slug = "_".join(page.relative_to(SITE).with_suffix("").parts).replace("-", "_")
        name = f"__dsgelab_site_{slug}__"
        module = types.ModuleType(name)
        sys.modules[name] = module
        ns = module.__dict__
        print(f"\n{page.relative_to(ROOT)}")
        for i, line, src in blocks(page):
            label = f"block {i} (line {line})"
            try:
                exec(compile(src, f"<site {label}>", "exec"), ns)
            except Exception:
                failures += 1
                print(f"  FAILED  {label}")
                print("    " + "\n    ".join(traceback.format_exc().strip().splitlines()[-4:]))
            else:
                print(f"  ok      {label}  ({len(src.splitlines())} lines)")
    print(f"\n{'every example ran' if failures == 0 else f'{failures} block(s) failed'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        raise SystemExit(main())
