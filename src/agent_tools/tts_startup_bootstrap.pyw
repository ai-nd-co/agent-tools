from __future__ import annotations

import sys
from pathlib import Path


def _main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] != "--module-root":
        return 2
    module_root = Path(sys.argv[2])
    if not module_root.is_absolute():
        return 2
    sys.path.insert(0, str(module_root))
    del sys.argv[1:3]
    from agent_tools.tts_startup import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
