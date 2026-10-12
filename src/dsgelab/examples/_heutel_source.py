"""Heutel's MATLAB replication files: download, check and translate them on the user's machine.

The scalar algebra of the port (steady-state equations and the coefficients of the linear
system) is a mechanical translation of Garth Heutel's MATLAB files. Those files carry no
licence, so dsgelab does not distribute the translation. :func:`fetch` downloads the public
copy, refuses any file whose SHA-256 differs from the one the port was validated on, and
writes the translation to a cache directory outside the package.

Only assignments are translated, by pattern, and the MATLAB text is never executed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import re
import urllib.request
import zipfile
from pathlib import Path
from types import ModuleType

#: The public copy, in Georgia State University's institutional repository
#: (scholarworks.gsu.edu, item 9a01aa68-89c0-4e66-b7bd-9f09606d259c).
SOURCE_URL = ("https://scholarworks.gsu.edu/server/api/core/bitstreams/"
              "9be302dc-8a22-440f-b0e8-ad22a5022b85/content")
ZIP_SHA256 = "7ea9b108c7d8b393c8214d4a958c9b7de5b3151180cd3e85ab65d28cd0a5c123"
#: The six files the port was validated on (validation/heutel/source-sha256.json).
FILE_SHA256 = {
    "SolveSystem.m": "07400e6cefdb1eecdb473551b313962f8a31780f20da2cbc0b661a75dce2543b",
    "SolveSystem_quota_AsymInfo.m": "f67fc842acfc14663f023094b84a69ed7fadeb64f164847c8239c7cebbfb218d",
    "SolveSystem_tax.m": "957109ebb951f4397f7f11e6f755c6b3d1fb7c9f663c3721b64c7c4b1be36df0",
    "SolveSystem_tax_AsymInfo.m": "1c9dc495e0d6b0c20ee925bd01d37363e82e3aca3472ce01d5fd0efadf1ca175",
    "steadystate.m": "16aa023ce4686ea0365cd83b16c6e4e15f2dd18d25c327572bb4bd50abb494fb",
    "steadystate_tax.m": "7b8f56be45d1ab85d49636a2c37aef0600f5df6233053da6419f6ceba96c7ba1",
}
MODULE_NAME = "_heutel_algebra.py"
#: SHA-256 of the translated module: the translation is deterministic, so any other
#: content in the cache (edited, or from another version) is refused rather than run.
ALGEBRA_SHA256 = "77dc504e35fd8f0f6c9db2e7d9ade40ae39848750a5fa2a373f3185edf3485c4"
PARAMS = ("beta", "delta", "rho", "alpha", "eta", "theta1", "theta2", "gamma", "phic",
          "d2", "d1", "d0", "erowcoef")


class HeutelSourceMissing(RuntimeError):
    """The translated algebra is not in the cache; run :func:`fetch` first."""


def cache_dir() -> Path:
    """``DSGELAB_HEUTEL_DIR`` if set, else ``~/.cache/dsgelab/heutel``."""
    env = os.environ.get("DSGELAB_HEUTEL_DIR")
    return Path(env) if env else Path.home() / ".cache" / "dsgelab" / "heutel"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_sources(source: str | Path | None) -> dict[str, str]:
    """The six .m files as text, checked against :data:`FILE_SHA256`."""
    files: dict[str, bytes] = {}
    if source is None or Path(source).suffix == ".zip":
        if source is None:
            with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
                data = response.read()
        else:
            data = Path(source).read_bytes()
        if _sha256(data) != ZIP_SHA256:
            raise ValueError("the downloaded archive does not match the expected SHA-256; "
                             "nothing was translated")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.namelist():
                name = Path(member).name
                if name in FILE_SHA256:
                    files[name] = archive.read(member)
    else:
        for name in FILE_SHA256:
            path = Path(source) / name
            if path.exists():
                files[name] = path.read_bytes()
    missing = sorted(set(FILE_SHA256) - set(files))
    if missing:
        raise ValueError(f"Heutel source files missing: {missing}")
    for name, data in files.items():
        if _sha256(data) != FILE_SHA256[name]:
            raise ValueError(f"{name} differs from the file the port was validated on; "
                             "nothing was translated")
    return {name: data.decode("latin-1") for name, data in files.items()}


def _logical(text: str) -> str:
    """Strip block and line comments and join MATLAB continuation lines."""
    s = re.sub(r"%\{.*?%\}", "", text, flags=re.S)
    s = "\n".join(line.split("%")[0] for line in s.splitlines())
    return re.sub(r"\.\.\.\s*\n\s*", "", s)


def _expr(s: str) -> str:
    return re.sub(r"\blambda\b", "lam", s.strip().replace("^", "**"))


def _index(v: str) -> str:
    if v == ":":
        return ":"
    if ":" in v:
        a, b = v.split(":")
        return f"{int(a) - 1}:{int(b)}"
    return str(int(v) - 1)


def _assignments(s: str) -> list[str]:
    lines = []
    for statement in s.split(";"):
        statement = statement.strip()
        if not statement:
            continue
        if "=" not in statement:
            raise ValueError(f"not an assignment: {statement!r}")
        left, right = (part.strip() for part in statement.split("=", 1))
        if left.startswith("h("):
            left = "h[" + ", ".join(_index(v) for v in left[2:-1].split(",")) + "]"
            if right.startswith("zeros("):
                right = "0.0"
        left = re.sub(r"X\((\d+)\)", lambda m: f"X[{int(m[1]) - 1}]", left)
        lines.append("    " + _expr(left) + " = " + _expr(right))
    return lines


def _unpack(names) -> str:
    return "".join(f"    {p} = p.{p}\n" for p in names)


def translate(sources: dict[str, str]) -> str:
    """Python source of the algebra module, from the checked MATLAB texts."""
    out = ('"""Scalar algebra translated from Garth Heutel\'s MATLAB replication files (2011).\n\n'
           "Generated on this machine by dsgelab.examples.heutel.fetch(); not part of dsgelab\n"
           "and not distributed with it. The equations keep the source order.\n"
           '"""\nimport numpy as np\n\n')
    for name, source, dim in [("planner_ss", "steadystate.m", 3), ("tax_ss", "steadystate_tax.m", 2)]:
        s = _logical(sources[source])
        s = s[s.index("k = guess(1)"):]
        s = re.sub(r"guess\((\d+)\)", lambda m: f"guess[{int(m[1]) - 1}]", s)
        out += f"def {name}(guess, p):\n" + _unpack(PARAMS) + f"    X = np.zeros({dim})\n"
        out += "\n".join(_assignments(s)) + "\n    return X\n\n"
    # Levels, derivatives and multipliers are shared by all the tax variants.
    s = _logical(sources["SolveSystem_tax.m"])
    s = s[s.index("i_ss = delta*k_ss"):s.index("neq = 14")]
    out += "def tax_levels(k_ss, e_ss, p):\n" + _unpack(PARAMS)
    out += ("\n".join(_assignments(s))
            + '\n    return {k[:-3]: v for k, v in locals().items() if k.endswith("_ss")}\n\n')
    for name, source in [("planner_matrix", "SolveSystem.m"), ("tax_matrix", "SolveSystem_tax.m")]:
        s = _logical(sources[source])
        s = s[s.index("h(1,"):s.index("condn =")]
        keys = ["c", "e", "k", "x", "mu", "y", "z", "i", "a"]
        if not name.startswith("planner"):
            keys += ["tau", "r", "lambda", "zeta", "omega"]
        out += f"def {name}(ss, p):\n" + _unpack(PARAMS)
        out += "".join(f'    {k}_ss = ss["{k}"]\n' for k in keys)
        out += f"    h = np.zeros(({len(keys)}, {len(keys) * 3}))\n"
        out += "\n".join(_assignments(s)) + "\n    return h\n\n"
    s = _logical(sources["SolveSystem_quota_AsymInfo.m"])
    start = s.index("h(9,:)")
    s = s[start:s.index("[b,rts", start)]
    out += "def quota_firm_rows(h, ss, p):\n" + _unpack(PARAMS)
    out += "".join(f'    {k}_ss = ss["{k}"]\n' for k in ["y", "k", "mu", "r"])
    out += "\n".join(_assignments(s)) + "\n    return h\n"
    return out


def fetch(source: str | Path | None = None, *, cache: str | Path | None = None) -> Path:
    """Download (or read) Heutel's files, check them and write the translated algebra.

    ``source`` is ``None`` to download the public copy, the path of that ``.zip``, or a
    directory holding the six ``.m`` files. Returns the path of the module written to
    ``cache`` (default :func:`cache_dir`).
    """
    code = translate(_read_sources(source))
    if _sha256(code.encode("utf-8")) != ALGEBRA_SHA256:
        raise RuntimeError("the translation differs from the validated one; nothing was written")
    target = Path(cache) if cache is not None else cache_dir()
    target.mkdir(parents=True, exist_ok=True)
    path = target / MODULE_NAME
    path.write_text(code, encoding="utf-8")
    _loaded.clear()
    return path


_loaded: dict[str, ModuleType] = {}


def load() -> ModuleType:
    """The translated algebra module, from the cache."""
    path = cache_dir() / MODULE_NAME
    key = str(path)
    if key not in _loaded:
        if not path.exists():
            raise HeutelSourceMissing(
                "Heutel's MATLAB files have not been translated on this machine. Run "
                "dsgelab.examples.heutel.fetch() once; it downloads the public copy and "
                "checks every file before translating it (see docs/heutel.md).")
        if _sha256(path.read_bytes()) != ALGEBRA_SHA256:
            raise HeutelSourceMissing(
                f"{path} is not the translation this version of dsgelab produces; "
                "run dsgelab.examples.heutel.fetch() again to rewrite it.")
        spec = importlib.util.spec_from_file_location("dsgelab_heutel_algebra", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _loaded[key] = module
    return _loaded[key]
