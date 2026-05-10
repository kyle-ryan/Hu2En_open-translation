"""Extract and normalize audio from video files via ffmpeg."""

import subprocess
from pathlib import Path


def extract_audio(video_path: Path, out_dir: Path) -> Path:
    """Extract audio from video to 16kHz mono WAV, required by Whisper and pyannote."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / (video_path.stem + ".wav")
    if wav_path.exists():
        return wav_path

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                   # no video
        "-ar", "16000",          # 16kHz
        "-ac", "1",              # mono
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    return wav_path
