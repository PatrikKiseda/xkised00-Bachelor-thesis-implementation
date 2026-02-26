"""
Author: Patrik Kiseda
File: scripts/check_qdrant_connectivity.py
Description: Very simple local connectivity check for Qdrant running in Docker on localhost.
"""

from __future__ import annotations
from qdrant_client import QdrantClient

import os
import sys


def main() -> int:
    qdrant_url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")

    try:
        client = QdrantClient(url=qdrant_url, timeout=3.0)
        collections = client.get_collections()
        count = len(collections.collections)
    except Exception as exc:  # pragma: no extra handling - simple CLI check
        print(f"[FAIL] Could not connect to Qdrant at {qdrant_url}")
        print(f"Reason: {exc}")
        print("Tip: start it with `make qdrant-up` first.")
        return 1

    print(f"[OK] Connected to Qdrant at {qdrant_url}")
    print(f"Collections found: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
