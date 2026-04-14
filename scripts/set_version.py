from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE_INIT = ROOT / "src" / "agent_tools" / "__init__.py"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/set_version.py <version>")

    version = sys.argv[1].strip()
    if not version:
        raise SystemExit("version must be non-empty")

    rewrite(
        PYPROJECT,
        r'(?m)^version = "[^"]+"$',
        f'version = "{version}"',
    )
    rewrite(
        PACKAGE_INIT,
        r'(?m)^__version__ = "[^"]+"$',
        f'__version__ = "{version}"',
    )
    return 0


def rewrite(path: Path, pattern: str, replacement: str) -> None:
    original = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, original)
    if count != 1:
        raise RuntimeError(f"expected exactly one replacement in {path}")
    path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
