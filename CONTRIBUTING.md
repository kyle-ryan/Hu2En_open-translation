# Contributing to Hu2En

Thank you for helping improve this project. Contributions fall into two categories:

1. **Translation corrections** — fixing errors in `<stem>.json` output files
2. **Code improvements** — bug fixes, new features, additional language support

---

## Translation corrections

The pipeline produces a `<stem>.json` file containing every segment with both the original Hungarian (`hu_text`) and the English translation (`en_text`). Errors in translation — wrong word choice, missed idioms, incorrect register — can be corrected directly in this file.

### Editing the JSON

Each segment looks like this:

```json
{
  "start": 3206.5,
  "end": 3211.2,
  "speaker": "SPEAKER_13",
  "hu_text": "Pontosan tudom és értem, hogy a nekem szavazott bizalom mekkora felelősség.",
  "en_text": "I know and understand exactly how great a responsibility the trust placed in me represents."
}
```

**Rules:**
- Edit `en_text` only — never alter `start`, `end`, `speaker`, or `hu_text`
- Preserve the speaker's register. A formal parliamentary address should read formally in English; a conversational interjection should not be elevated to formal prose
- Keep punctuation consistent with surrounding segments
- Do not split or merge segments — one JSON object stays one subtitle line

### Regenerating the SRT after edits

```bash
.venv/bin/vidtrans video.mkv --skip-transcribe --skip-translate
```

This loads the corrected `.json` and writes a fresh `.en.srt` without running any models.

### Submitting a correction PR

1. Fork the repository
2. Create a branch: `git checkout -b fix/translation-segment-1234`
3. Edit the relevant `.json` file
4. Regenerate and verify the `.srt` plays correctly with the video
5. Open a pull request with:
   - The segment index or timestamp range affected
   - The original `en_text`
   - The corrected `en_text`
   - A brief reason (e.g. *"idiom mistranslated"*, *"proper noun misspelled"*)

**Do not commit** the `.wav`, `.mp4`, `.mkv`, or `.whisper.json` files. See `.gitignore`.

---

## Code contributions

### Setting up for development

```bash
git clone https://github.com/kyle-ryan/Hu2En_open-translation.git
cd Hu2En_open-translation
uv sync
```

### Project layout

```
src/vidtrans/
├── audio.py        ffmpeg audio extraction
├── transcribe.py   mlx-whisper STT + pyannote diarisation
├── translate.py    Qwen3-32B HU→EN translation
├── formats.py      JSON ↔ SRT conversion
└── pipeline.py     CLI entrypoint
```

### Before opening a PR

- Run the pipeline end-to-end on a short test clip to confirm nothing is broken
- Keep each PR focused on one concern — don't mix translation fixes with code changes
- Update `README.md` if you add or change a CLI flag

### Adding support for another language pair

1. The `--language` flag is passed directly to Whisper — it accepts any [Whisper-supported language code](https://github.com/openai/whisper#available-models-and-languages)
2. Update the prompt in `translate.py → _build_prompt()` to reflect the new source language
3. Update the `--domain` default if it no longer applies
4. Add a note to `README.md`

### Reporting issues

Use the **Pipeline bug report** issue template on GitHub — it prompts for chip, macOS version, command, and full error output. Filling it in completely gets you a faster response.

---

## When to open an issue vs. a PR

| Situation | Action |
|-----------|--------|
| You found a translation error and have a fix | Open a PR directly |
| You found a translation error but are unsure of the correct English | Open a **Translation correction** issue |
| The pipeline crashed or produced wrong output | Open a **Pipeline bug report** issue |
| You want to add a feature or new language | Open an issue first to discuss approach |

---

## CI — what runs on every PR touching `subtitles/`

A GitHub Actions workflow (`.github/workflows/validate.yml`) runs automatically and checks:

1. **JSON syntax** — all `.json` files in `subtitles/` parse without error
2. **Field completeness** — every segment has `hu_text`, `en_text`, `start`, `end`
3. **Timestamp integrity** — no SRT timestamps differ from the values in the paired `.json` (guards against accidental edits to timing)

To run the same checks locally before pushing:

```bash
python .github/scripts/validate_subtitles.py
```

---

## Code of conduct

Be direct, be constructive, assume good faith. Translation is subjective — disagreements about phrasing should cite the source text and explain the reasoning.
