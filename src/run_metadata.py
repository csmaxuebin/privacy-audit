"""Run metadata tracking for experiment reproducibility.

Appends structured metadata entries to a JSONL file (one JSON object per line).
Each entry is automatically enriched with timestamp and git commit hash.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

METADATA_PATH = "reports/run_metadata.jsonl"


def get_git_commit() -> str:
    """Return current git HEAD commit hash, or 'unknown' if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def append_metadata(entry: dict, path: str = METADATA_PATH):
    """Append a single metadata entry to the JSONL file.

    Automatically adds 'timestamp' (UTC ISO format) and 'commit' fields.
    Creates parent directories and file if they don't exist.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    entry["commit"] = get_git_commit()
    with open(p, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_metadata(path: str = METADATA_PATH) -> list:
    """Read all metadata entries from the JSONL file.

    Skips corrupted lines (invalid JSON) gracefully.
    Returns empty list if file does not exist.
    """
    p = Path(path)
    if not p.exists():
        return []
    entries = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip corrupted lines
    return entries
