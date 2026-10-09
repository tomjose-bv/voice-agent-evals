"""Fail if any term from the local, git-ignored .employer-terms file appears in tracked files.

Usage: python scripts/check_terms.py
The terms file holds one case-insensitive term per line. It is never committed,
so the terms themselves never end up in the repository.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    terms_file = ROOT / ".employer-terms"
    if not terms_file.exists():
        print("no .employer-terms file; nothing to check")
        return 0
    terms = [t.strip().lower() for t in terms_file.read_text(encoding="utf-8").splitlines() if t.strip()]
    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.splitlines()
    hits = 0
    for rel in files:
        p = ROOT / rel
        try:
            text = p.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        for t in terms:
            if t in text:
                hits += 1
                print(f"{rel}: contains a private term (#{terms.index(t) + 1})")
    print("OK: no private terms found" if not hits else f"FAIL: {hits} hit(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
