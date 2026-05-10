#!/usr/bin/env python3
"""CI validation for subtitle files in subtitles/.

Checks:
1. Every segments JSON has hu_text and en_text on all entries
2. SRT segment count matches the segments JSON for the same stem
3. No SRT timestamps differ from what the JSON would produce
   (guards against contributors accidentally editing timestamps)

Run locally:  python .github/scripts/validate_subtitles.py
"""

import json
import re
import sys
from pathlib import Path

SUBTITLES = Path("subtitles")
errors: list[str] = []


def ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ── 1. Validate each segments JSON ───────────────────────────────────────────
for json_path in sorted(SUBTITLES.glob("*.json")):
    if "whisper" in json_path.name:
        continue  # whisper checkpoint — different schema, skip deep check
    print(f"Checking {json_path.name} ...")
    try:
        segments = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        errors.append(f"{json_path.name}: invalid JSON — {e}")
        continue

    for i, seg in enumerate(segments):
        if "hu_text" not in seg:
            errors.append(f"{json_path.name} segment {i}: missing hu_text")
        if "en_text" not in seg:
            errors.append(f"{json_path.name} segment {i}: missing en_text")
        if "start" not in seg or "end" not in seg:
            errors.append(f"{json_path.name} segment {i}: missing start/end")

    # ── 2+3. Cross-check against the paired SRT ──────────────────────────────
    stem = json_path.stem  # e.g. HU_PM_election_transition_2026-05-09
    srt_path = SUBTITLES / f"{stem}.en.srt"
    if not srt_path.exists():
        # No SRT to compare — that's fine, just skip cross-check
        continue

    srt_text = srt_path.read_text(encoding="utf-8")
    # Parse SRT timestamp lines
    srt_timestamps = re.findall(
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", srt_text
    )

    if len(srt_timestamps) != len(segments):
        errors.append(
            f"{srt_path.name}: {len(srt_timestamps)} SRT entries vs "
            f"{len(segments)} JSON segments — counts must match"
        )
        continue

    for i, (seg, (srt_start, srt_end)) in enumerate(
        zip(segments, srt_timestamps), 1
    ):
        expected_start = ts(seg["start"])
        expected_end   = ts(seg["end"])
        if srt_start != expected_start or srt_end != expected_end:
            errors.append(
                f"{srt_path.name} entry {i}: timestamp changed.\n"
                f"  JSON says : {expected_start} --> {expected_end}\n"
                f"  SRT has   : {srt_start} --> {srt_end}\n"
                f"  Timestamps must not be edited — only the text line."
            )


# ── Report ────────────────────────────────────────────────────────────────────
if errors:
    print("\n❌ Validation failed:\n")
    for e in errors:
        print(f"  • {e}")
    sys.exit(1)
else:
    print("\n✅ All subtitle files valid.")
