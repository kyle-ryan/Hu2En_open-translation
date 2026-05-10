# Hu2En — Hungarian Video → English Subtitles (fully local)

A fully local pipeline for transcribing Hungarian-language video, labelling speakers, and producing accurate English subtitles. No data leaves your machine.

```
AV1 video (Hungarian)
  │
  ├─ ffmpeg           extract 16 kHz mono WAV
  ├─ mlx-whisper      speech-to-text  →  <stem>.whisper.json
  ├─ pyannote         speaker diarization
  ├─ Qwen3-32B        Hungarian → English translation
  └─ SRT writer       <stem>.en.srt  (+ optional <stem>.hu.srt)
```

All heavy models run locally via **MLX** (Apple Silicon) and **PyTorch MPS/CPU**.

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| Apple Silicon Mac | M1 Max or later recommended |
| 64 GB unified memory | Qwen3-32B-8bit alone needs ~34 GB |
| macOS 14+ | MLX Metal backend |
| Python 3.11 | Managed via `uv` |
| [uv](https://docs.astral.sh/uv/) | `brew install uv` |
| ffmpeg | `brew install ffmpeg` |
| HuggingFace account | Free — token needed for pyannote gated model |

---

## One-time setup

### 1 — Clone and install

```bash
git clone https://github.com/kyle-ryan/Hu2En_open-translatation.git
cd Hu2En_open-translatation
uv sync          # creates .venv and installs all dependencies
```

### 2 — Accept the pyannote model licence

Visit **https://huggingface.co/pyannote/speaker-diarization-community-1** and click *Agree and access repository*. A free HuggingFace account is required.

### 3 — Set your HuggingFace token

```bash
export HF_TOKEN=hf_your_token_here
```

Add this to your `~/.zshrc` or `~/.bash_profile` to make it permanent.

> **Security note:** never paste your token into code or commit it to git. The pipeline reads it exclusively from the environment variable.

---

## Usage

```bash
# Full pipeline — transcribe, diarise, translate
.venv/bin/vidtrans path/to/video.mkv

# Also write a Hungarian SRT
.venv/bin/vidtrans video.mkv --hu-srt

# Specify content type for better translation quality
.venv/bin/vidtrans video.mkv --domain "political documentary interview"

# Custom output directory
.venv/bin/vidtrans video.mkv --out ~/Desktop/output/
```

### Resume flags — restart a failed or interrupted run

Each stage writes a checkpoint. If something fails mid-run you don't have to start over.

```bash
# Whisper finished but pyannote crashed → resume from diarisation
.venv/bin/vidtrans video.mkv --skip-whisper

# Both STT stages done → jump straight to translation
.venv/bin/vidtrans video.mkv --skip-transcribe

# Skip translation entirely — write SRT in source language only
.venv/bin/vidtrans video.mkv --skip-translate
```

### Stage checkpoints

| File | Created after |
|------|---------------|
| `tmp/<stem>.wav` | ffmpeg extraction |
| `<stem>.whisper.json` | mlx-whisper transcription |
| `<stem>.json` | pyannote diarisation + merge |
| `<stem>.en.srt` | Qwen3 translation |

---

## Attaching subtitles to your video

### Soft subtitles (recommended — toggleable in any player)

```bash
# Add as a subtitle track — video is not re-encoded
ffmpeg -i video.mkv -i video.en.srt \
  -c copy -c:s srt \
  -metadata:s:s:0 language=eng \
  video_with_subs.mkv
```

To include both languages:

```bash
ffmpeg -i video.mkv -i video.en.srt -i video.hu.srt \
  -c copy -c:s srt \
  -metadata:s:s:0 language=eng -metadata:s:s:0 title="English" \
  -metadata:s:s:1 language=hun -metadata:s:s:1 title="Hungarian" \
  video_with_subs.mkv
```

### Player quick-start (no encoding needed)

**VLC** — drag the `.srt` file onto the playing video, or *Subtitle → Add Subtitle File…*

**mpv** — place the `.srt` file in the same directory with the same base name; mpv loads it automatically:
```bash
mpv video.mkv   # picks up video.en.srt automatically
```

**IINA (macOS)** — *File → Open Subtitle…* or drag-and-drop onto the window.

### Hard subtitles (burned into picture — not recommended unless required)

```bash
ffmpeg -i video.mkv \
  -vf "subtitles=video.en.srt:force_style='FontName=Arial,FontSize=24'" \
  -c:a copy \
  video_hardsub.mp4
```

---

## Models used

| Stage | Model | Runtime |
|-------|-------|---------|
| Speech-to-text | [mlx-community/whisper-large-v3-mlx](https://huggingface.co/mlx-community/whisper-large-v3-mlx) | MLX (Metal) |
| Speaker diarisation | [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | PyTorch MPS |
| Translation | [mlx-community/Qwen3-32B-8bit](https://huggingface.co/mlx-community/Qwen3-32B-8bit) | MLX (Metal) |

All models are downloaded automatically on first run via HuggingFace Hub and cached locally.

---

## Translation quality notes

- The `--domain` flag provides context to the translation model. Always set it to match your content — it meaningfully improves register and terminology.
- Speaker labels (e.g. `SPEAKER_00`) are passed to the model so it can maintain consistent voice across a speaker's turns.
- Segments that Whisper hallucinates repeatedly (a known issue on music/silence) are deduplicated before translation and copied rather than re-inferred.
- Translations can be corrected manually by editing `<stem>.json` and re-running `--skip-transcribe`.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to submit translation corrections, code improvements, and new language support.

---

## Licence

Apache 2.0 — see [LICENSE](LICENSE).

The models used by this pipeline have their own licences and terms of use. Review them before use in commercial or derivative works.
