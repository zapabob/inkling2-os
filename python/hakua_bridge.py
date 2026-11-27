"""Hakua-memory bridge for Inkling2-OS (managed via uv; logging-only per MILSPEC)."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [HakuaBridge] %(message)s",
)
LOGGER = logging.getLogger("hakua_bridge")


def probe_hakua_memory() -> int:
    """Verify hakua-memory package installation and report version & exports."""
    try:
        version = importlib.metadata.version("hakua-memory")
    except importlib.metadata.PackageNotFoundError:
        LOGGER.error("hakua-memory is not installed in the active uv environment")
        return 1

    mod = None
    for candidate in ("hakua_memory", "hakua"):
        try:
            mod = importlib.import_module(candidate)
            break
        except ImportError:
            continue

    exports = sorted(a for a in dir(mod) if not a.startswith("_"))[:12] if mod is not None else []
    LOGGER.info("hakua-memory v%s ready (module=%s, exports=%s)", version, getattr(mod, "__name__", "n/a"), exports)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inkling2-OS hakua-memory bridge")
    parser.add_argument("--probe", action="store_true", help="Verify hakua-memory availability")
    args = parser.parse_args()
    if args.probe:
        return probe_hakua_memory()
    return probe_hakua_memory()


if __name__ == "__main__":
    sys.exit(main())
