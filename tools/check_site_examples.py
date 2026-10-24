#!/usr/bin/env python
"""Runs the code blocks of site/**/*.html, in both languages.

An example that does not run misleads the reader, so every block is executed.
Each page has its own namespace, and within a page the blocks share names, in
order. The blocks run in a temporary directory, because the ones that export
models write files, and those must not end up in the repository.

The page also shows what each block produces, in a ``<pre class="output">`` right
after it: what the block prints, and the value of every expression that stands
alone on a line, as an interactive session would show it. The output is not typed
by hand: ``--write`` runs the blocks and writes it, and the check fails when the
output on the page is not what the code prints. Numbers are compared with a
tolerance, because the last digits of a residual at machine precision differ
between platforms.

    python tools/check_site_examples.py            # run, and compare the outputs
    python tools/check_site_examples.py --write    # run, and write the outputs
"""
from __future__ import annotations

import ast
import contextlib
import html
import io
import numbers
import os
import re
import sys
import tempfile
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
BLOCK = re.compile(r'<pre><code(?P<attr>[^>]*)>(?P<src>.*?)</code></pre>'
                   r'(?P<out>\s*<pre class="output"[^>]*><code>(?P<shown>.*?)</code></pre>)?', re.S)
LABEL = {"en": "Output", "pt-br": "Saída"}
EXPR = "\x00"           # marks, inside the captured text, the line that names an expression
# (not \x1e or \x1c: str.splitlines splits on them)
MAX_LINES = 30
MAX_ROWS = 12           # a long table shows its head and tail
MAX_COLUMNS = 8
NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def blocks(page: Path):
    """The page's executable blocks, in order, with HTML escapes undone."""
    text = page.read_text(encoding="utf-8")
    for i, m in enumerate(BLOCK.finditer(text), start=1):
        if "plain" in m.group("attr"):
            continue
        line = text[:m.start()].count("\n") + 1
        yield i, line, html.unescape(m.group("src")).strip()


def shown(page: Path) -> dict[int, str]:
    """The output the page shows under each executable block, as plain text."""
    text = page.read_text(encoding="utf-8")
    return {i: _plain(m.group("shown") or "") for i, m in enumerate(BLOCK.finditer(text), start=1)
            if "plain" not in m.group("attr")}


def _plain(fragment: str) -> str:
    return html.unescape(re.sub(r"</?span[^>]*>", "", fragment)).strip("\n")


def _g(x: float) -> str:
    return f"{x:.4g}"


def show(value) -> str:
    """A value as the page prints it: four significant digits, no dtype lines, no addresses."""
    import numpy as np
    try:
        import pandas as pd
        frame, series = pd.DataFrame, pd.Series
    except ImportError:                       # dsgelab does not depend on pandas
        frame = series = ()
    if isinstance(value, (bool, np.bool_)):
        text = str(bool(value))
    elif isinstance(value, numbers.Integral):
        text = str(int(value))
    elif isinstance(value, numbers.Real):
        text = _g(float(value))
    elif frame and isinstance(value, frame):
        text = value.to_string(float_format=_g, max_rows=MAX_ROWS, max_cols=MAX_COLUMNS)
    elif series and isinstance(value, series):
        text = value.to_string(float_format=_g, max_rows=MAX_ROWS)
    elif isinstance(value, np.ndarray):
        text = np.array2string(value, max_line_width=88, threshold=60,
                               formatter={"float_kind": _g})
    elif isinstance(value, tuple):
        inner = ", ".join(show(v) for v in value)
        text = f"({inner}{',' if len(value) == 1 else ''})"
    elif isinstance(value, list):
        text = "[" + ", ".join(show(v) for v in value) + "]"
    elif isinstance(value, dict):
        items = [f"{k!r}: {show(v)}" for k, v in value.items()]
        one = "{" + ", ".join(items) + "}"
        text = one if len(one) <= 88 else "{\n" + ",\n".join(f"  {s}" for s in items) + "\n}"
    else:
        text = re.sub(r" at 0x[0-9a-fA-F]+", "", repr(value))
    lines = text.splitlines()
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES] + ["..."]
    return "\n".join(lines)


def execute(src: str, ns: dict, label: str) -> str:
    """Run one block statement by statement; return what it printed and what it displayed."""
    tree = ast.parse(src, filename=f"<site {label}>")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        for node in tree.body:
            if isinstance(node, ast.Expr):
                value = eval(compile(ast.Expression(node.value), f"<site {label}>", "eval"), ns)
                if value is not None:
                    buffer.write(f"{EXPR}{ast.get_source_segment(src, node)}\n{show(value)}\n")
            else:
                exec(compile(ast.Module([node], type_ignores=[]), f"<site {label}>", "exec"), ns)
    return buffer.getvalue().rstrip("\n")


def same(a: str, b: str) -> bool:
    """Equal text, and numbers equal to a tolerance; runs of white space count as one."""
    a, b = " ".join(a.split()), " ".join(b.split())
    if NUMBER.split(a) != NUMBER.split(b):
        return False
    for x, y in zip(NUMBER.findall(a), NUMBER.findall(b)):
        x, y = float(x), float(y)
        if abs(x - y) > 1e-8 and abs(x - y) > 2e-3 * max(abs(x), abs(y)):
            return False
    return True


def run(page: Path, verbose: bool = True, outputs: dict | None = None) -> int:
    """Execute one page's blocks in a fresh namespace; return the number that failed.

    A block fails when it raises, or when the output under it on the page is not what it
    prints. With ``outputs``, the captured output of each block is stored there instead of
    compared, for ``--write``.
    """
    slug = "_".join(page.relative_to(SITE).with_suffix("").parts).replace("-", "_")
    name = f"__dsgelab_site_{slug}__"
    module = types.ModuleType(name)
    sys.modules[name] = module
    ns = module.__dict__
    on_page = shown(page)
    failures = 0
    for i, line, src in blocks(page):
        label = f"block {i} (line {line})"
        try:
            got = execute(src, ns, label)
        except Exception:
            failures += 1
            if verbose:
                print(f"  FAILED  {label}")
                print("    " + "\n    ".join(traceback.format_exc().strip().splitlines()[-4:]))
                print("    --- code ---")
                print("    " + "\n    ".join(src.splitlines()[:6]))
            continue
        if outputs is not None:
            outputs[i] = got
        elif not same(got.replace(EXPR, ""), on_page[i]):
            failures += 1
            if verbose:
                print(f"  STALE   {label}: the output on the page is not what the code prints;"
                      " run tools/check_site_examples.py --write")
                print("    --- printed ---")
                print("    " + "\n    ".join(got.replace(EXPR, "").splitlines()[:8]))
            continue
        if verbose:
            print(f"  ok      {label}  ({len(src.splitlines())} lines)")
    return failures


def render(text: str, label: str) -> str:
    lines = [f'<span class="out-expr">{html.escape(s[1:], quote=False)}</span>'
             if s.startswith(EXPR) else html.escape(s, quote=False) for s in text.splitlines()]
    return f'<pre class="output" data-label="{label}"><code>' + "\n".join(lines) + "</code></pre>"


def write(page: Path, outputs: dict, label: str) -> None:
    """Put each block's output right under it, replacing the one already there."""
    text = page.read_text(encoding="utf-8")
    count = iter(range(1, 10**6))

    def replace(m):
        i = next(count)
        code = text[m.start():m.end("src") + len("</code></pre>")]
        if "plain" in m.group("attr") or not outputs.get(i):
            return code
        return code + "\n" + render(outputs[i], label)

    page.write_text(BLOCK.sub(replace, text), encoding="utf-8")


def main(argv: list[str]) -> int:
    writing = "--write" in argv
    pages = sorted(SITE.rglob("*.html"))
    failures = 0
    for page in pages:
        print(f"\n{page.relative_to(ROOT)}")
        outputs = {} if writing else None
        failed = run(page, outputs=outputs)
        failures += failed
        if writing and not failed:
            lang = "pt-br" if "pt-br" in page.relative_to(SITE).parts else "en"
            write(page, outputs, LABEL[lang])
            print(f"  wrote the outputs of {len(outputs)} block(s)")
    print(f"\n{'every example ran' if failures == 0 else f'{failures} block(s) failed'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        raise SystemExit(main(sys.argv[1:]))
