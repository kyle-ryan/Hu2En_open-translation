# Hu2En — Hungarian Video → English Subtitles

English subtitles for Hungarian political video, generated locally and open for community correction.

---

## Just here for the subtitles?

Download the SRT from the [`subtitles/`](subtitles/) folder and load it in your player.

**VLC** — drag the `.srt` onto the playing video, or *Subtitle → Add Subtitle File…*

**mpv** — put the `.srt` in the same folder as the video with the same base name — it loads automatically:
```
video.mkv
video.en.srt   ← mpv picks this up on its own
```

**IINA (macOS)** — *File → Open Subtitle…* or drag-and-drop onto the window.

Want to correct a mistranslation? See [CONTRIBUTING.md](CONTRIBUTING.md) — no coding needed.

---

## Want to generate subtitles yourself?

The pipeline runs fully locally on Apple Silicon — no data leaves your machine.

### What you need

- Apple Silicon Mac (M1 Max or later — 64 GB unified memory recommended)
- [Homebrew](https://brew.sh)
- A free [HuggingFace](https://huggingface.co) account

### Setup

```bash
# Install dependencies
brew install uv ffmpeg

# Clone and install
git clone https://github.com/kyle-ryan/Hu2En_open-translation.git
cd Hu2En_open-translation
uv sync
```

Then accept the speaker diarisation model licence at **https://huggingface.co/pyannote/speaker-diarization-community-1** (free account required), and set your HuggingFace token:

```bash
export HF_TOKEN=hf_your_token_here   # add to ~/.zshrc to make permanent
```

### Run

```bash
.venv/bin/vidtrans path/to/video.mkv
```

This produces `<video name>.en.srt` next to the video. For best translation results, tell it what kind of content it is:

```bash
.venv/bin/vidtrans video.mkv --domain "parliamentary debate"
```

### Embed subtitles permanently with ffmpeg

```bash
ffmpeg -i video.mkv -i video.en.srt \
  -c copy -c:s srt -metadata:s:s:0 language=eng \
  video_with_subs.mkv
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — corrections to existing translations, new subtitle files, and code improvements are all welcome.

---

## Licence

Apache 2.0 — see [LICENSE](LICENSE).
The models used by this pipeline carry their own licences — review them before commercial use.

---

<details>
<summary>Pipeline and technical details</summary>

### How it works

```
AV1 video (Hungarian)
  │
  ├─ ffmpeg           extract 16 kHz mono WAV
  ├─ mlx-whisper      speech-to-text  →  <stem>.whisper.json
  ├─ pyannote         speaker diarization
  ├─ Qwen3-32B        Hungarian → English translation
  └─ SRT writer       <stem>.en.srt  (+ optional <stem>.hu.srt)
```

All models run locally via MLX (Metal) and PyTorch MPS/CPU.

### Models

| Stage | Model |
|-------|-------|
| Speech-to-text | [mlx-community/whisper-large-v3-mlx](https://huggingface.co/mlx-community/whisper-large-v3-mlx) |
| Speaker diarisation | [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) |
| Translation | [mlx-community/Qwen3-32B-8bit](https://huggingface.co/mlx-community/Qwen3-32B-8bit) |

Models download automatically on first run and cache locally.

### Resume flags

Each stage writes a checkpoint — if something fails you don't have to start from scratch.

```bash
.venv/bin/vidtrans video.mkv --skip-whisper      # Whisper done, re-run diarisation
.venv/bin/vidtrans video.mkv --skip-transcribe   # Jump straight to translation
.venv/bin/vidtrans video.mkv --skip-translate    # Write SRT in source language only
.venv/bin/vidtrans video.mkv --hu-srt            # Also output a Hungarian SRT
```

### Translation quality

- `--domain` gives the translation model context — always set it to match your content
- Speaker labels are passed through so the model maintains consistent voice per speaker
- Repeated hallucinated segments (common on silence/music) are deduplicated before translation

</details>
