## What does this PR change?

<!-- Brief description — e.g. "Fix 12 translation errors in segments 1842–1860" -->

---

## Translation corrections checklist

_Complete this section if you edited any subtitle or JSON files._

- [ ] I edited `en_text` fields only — `start`, `end`, `speaker`, and `hu_text` are unchanged
- [ ] I did not alter any timestamps in the `.srt` file
- [ ] I tested the corrected `.srt` in VLC or mpv against the source video
- [ ] Reading speed and timing feel natural at the corrected segments

**Segments changed** (list sequence numbers or timestamp ranges):

| # | Original `en_text` | Corrected `en_text` | Reason |
|---|-------------------|---------------------|--------|
|   |                   |                     |        |

---

## Code changes checklist

_Complete this section if you changed any `.py` files._

- [ ] I ran the pipeline end-to-end on a short test clip
- [ ] I updated `README.md` if I added or changed a CLI flag
- [ ] My change is limited to one concern (no mixing translation fixes with code changes)
