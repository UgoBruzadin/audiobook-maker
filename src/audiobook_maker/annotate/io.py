"""I/O for annotated scripts: save/load JSON."""

import json
from pathlib import Path

from .annotator import ScriptEntry


def save_script(entries: list[ScriptEntry], output_path: str | Path):
    """Save annotated script to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [e.to_dict() for e in entries]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Script saved: {output_path} ({len(entries)} entries)")


def load_script(path: str | Path) -> list[ScriptEntry]:
    """Load annotated script from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [ScriptEntry(**entry) for entry in data]
