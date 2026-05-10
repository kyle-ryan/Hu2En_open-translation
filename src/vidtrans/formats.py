"""JSON intermediate format and SRT output."""

import json
from pathlib import Path


def save_json(segments: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(segments: list[dict], text_field: str = "en_text") -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        text = seg.get(text_field) or seg.get("hu_text", "")
        speaker = seg.get("speaker", "")
        display = f"[{speaker}] {text}" if speaker else text
        lines += [
            str(i),
            f"{_ts(seg['start'])} --> {_ts(seg['end'])}",
            display,
            "",
        ]
    return "\n".join(lines)


def save_srt(segments: list[dict], path: Path, text_field: str = "en_text") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_srt(segments, text_field), encoding="utf-8")
