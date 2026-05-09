"""
Author: Patrik Kiseda
File: scripts/check_qdrant_connectivity.py
Description: Connectivity check that reuses the qdrant_store wrapper.
"""

from __future__ import annotations
import sys
from pathlib import Path


# make sure we can import from src.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from app.core.settings import get_settings
from app.storage.qdrant_store import QdrantStore

def main() -> int:
    """Check Qdrant reachability from configured settings.

    Returns:
        Process exit code.
    """
    settings = get_settings()
    store = QdrantStore.from_settings(settings)
    status = store.check_connection()

    if not status.reachable:
        print(f"[FAIL] Could not connect to Qdrant at {settings.qdrant_url}")
        print(f"Reason: {status.error}")
        print("Tip: start it with `make qdrant-up` first.")
        return 1

    print(f"[OK] Connected to Qdrant at {settings.qdrant_url}")
    return 0


# __main__ guard: allows file execution as a standalone script.
if __name__ == "__main__":
    sys.exit(main())
