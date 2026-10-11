#!/usr/bin/env python
"""Executa os blocos de código de site/*.html.

Um exemplo que não roda é pior do que exemplo nenhum. Cada página tem o seu
espaço de nomes, e dentro da página os blocos compartilham nomes, na ordem.

    python tools/check_site_examples.py
"""
from __future__ import annotations

import html
import re
import sys
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
    pages = sorted(SITE.glob("*.html"))
    failures = 0
    for page in pages:
        name = f"__dsgelab_site_{page.stem}__"
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
    raise SystemExit(main())
