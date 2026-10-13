#!/usr/bin/env python
"""Executa os blocos de código de site/**/*.html, nas duas línguas.

Um exemplo que não roda é pior do que exemplo nenhum. Cada página tem o seu
espaço de nomes, e dentro da página os blocos compartilham nomes, na ordem.
Os blocos rodam num diretório temporário: os que exportam modelos gravam
arquivos, e eles não devem ir parar no repositório.

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
