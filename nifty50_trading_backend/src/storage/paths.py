import glob
import os
from datetime import datetime
from typing import Optional, List


# PUBLIC_INTERFACE
def ensure_dir(path: str) -> str:
    """Ensure a directory exists, creating it if absent, and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


# PUBLIC_INTERFACE
def timestamped_filename(prefix: str, ext: str = "csv") -> str:
    """Create a timestamped filename like prefix_YYYYmmdd_HHMMSS.ext"""
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_prefix = "".join([c if c.isalnum() or c in ("_", "-") else "_" for c in prefix])
    return f"{safe_prefix}_{now}.{ext}"


# PUBLIC_INTERFACE
def latest_file(directory: str, patterns: Optional[List[str]] = None) -> Optional[str]:
    """Return the most recently modified file in directory matching any of patterns.

    Args:
        directory: Directory to search.
        patterns: List of glob wildcard patterns (e.g. ['*.csv', '*.json']). If None, uses ['*'].

    Returns:
        Path to the most recently modified file, or None if not found.
    """
    if patterns is None:
        patterns = ["*"]
    candidates: List[str] = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(directory, pat)))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


# PUBLIC_INTERFACE
def join(*parts: str) -> str:
    """Join path parts with OS-specific separator."""
    return os.path.join(*parts)
