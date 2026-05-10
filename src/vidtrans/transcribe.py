"""STT via mlx-whisper large-v3 + speaker diarization via pyannote.audio.

Checkpoints
-----------
<stem>.whisper.json  — raw Whisper output, written immediately after STT so
                       diarization can be re-run independently if needed.
<stem>.json          — merged segments (speaker + timestamps + hu_text),
                       written by pipeline.py after both stages complete.

Models are loaded, used, and explicitly freed one at a time to avoid
holding both an MLX model and a PyTorch/MPS model in memory simultaneously.

Apple Silicon / pyannote notes
------------------------------
MPS is not used for pyannote: PyTorch MPS is missing FFT operators that
pyannote's segmentation model requires (aten::_fft_r2c), so inference runs
on CPU.  On M-series chips this is still fast because we:
  - pin torch threads to the performance-core count
  - run under torch.inference_mode() (no grad bookkeeping)
  - pass batch_size tuned for the unified-memory bandwidth
"""

import gc
import json
import os
from pathlib import Path
from typing import Any

WHISPER_MODEL  = "mlx-community/whisper-large-v3-mlx"
DIARIZE_MODEL  = "pyannote/speaker-diarization-community-1"

# M1 Max has 8 performance + 2 efficiency cores (10 logical total).
# Reserve 2 for the OS/efficiency; give the rest to torch.
_TORCH_THREADS = max(4, (os.cpu_count() or 10) - 2)

# Batch sizes for segmentation and embedding.
# Each 10 s segmentation chunk @ 16 kHz float32 ≈ 640 KB.
# batch_size=64 → ~41 MB/batch; well within M1 Max's 64 GB unified memory.
# Embedding batches are speaker-sized, much smaller.
_SEGMENTATION_BATCH = 64
_EMBEDDING_BATCH    = 64


# ── Memory helpers ────────────────────────────────────────────────────────────

def _free_mlx_memory() -> None:
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception:
        pass
    gc.collect()


def _free_torch_memory() -> None:
    try:
        import torch
        # MPS is not used for pyannote, but flush anyway in case torch
        # allocated anything there during import.
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    gc.collect()


# ── Stage 1a: Whisper ─────────────────────────────────────────────────────────

def _transcribe(wav_path: Path, language: str) -> dict[str, Any]:
    import mlx_whisper

    print(f"[transcribe] Running mlx-whisper {WHISPER_MODEL} ...")
    try:
        return mlx_whisper.transcribe(
            str(wav_path),
            path_or_hf_repo=WHISPER_MODEL,
            word_timestamps=True,
            language=language,
            verbose=False,
        )
    finally:
        print("[transcribe] Freeing Whisper from Metal cache ...")
        _free_mlx_memory()


def save_whisper_checkpoint(whisper_result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(whisper_result, ensure_ascii=False, indent=2), encoding="utf-8")


def load_whisper_checkpoint(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ── Stage 1b: pyannote diarization ───────────────────────────────────────────

def _load_waveform(wav_path: Path) -> dict:
    """Load the full waveform into unified memory once.

    Passing a pre-loaded tensor dict to pyannote eliminates the ~21 000 disk
    seeks it would otherwise make while sliding over a 6-hour file.  The
    waveform stays on CPU; pyannote slices and moves chunks to MPS internally.
    """
    import torchaudio
    print(f"[diarize] Pre-loading waveform into memory ({wav_path.stat().st_size / 1e6:.0f} MB) ...")
    waveform, sample_rate = torchaudio.load(str(wav_path))
    return {"waveform": waveform, "sample_rate": sample_rate}


def _diarize(wav_path: Path, hf_token: str) -> Any:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    import torch

    # Pin torch threads — performance cores only, leave efficiency cores for OS.
    torch.set_num_threads(_TORCH_THREADS)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[diarize] Running {DIARIZE_MODEL} on {device.type.upper()} "
          f"({_TORCH_THREADS} threads, seg_batch={_SEGMENTATION_BATCH} "
          f"emb_batch={_EMBEDDING_BATCH}) ...")

    # Pre-load before the model occupies memory, so both fit comfortably.
    audio_in = _load_waveform(wav_path)

    pipeline = Pipeline.from_pretrained(DIARIZE_MODEL, token=hf_token)
    pipeline = pipeline.to(device)

    # batch_size is a property on the pipeline object, not a call argument
    pipeline.segmentation_batch_size = _SEGMENTATION_BATCH
    pipeline.embedding_batch_size    = _EMBEDDING_BATCH

    try:
        with torch.inference_mode(), ProgressHook() as hook:
            return pipeline(audio_in, hook=hook)
    finally:
        print("[diarize] Freeing pyannote ...")
        del pipeline, audio_in
        _free_torch_memory()


# ── Merge ─────────────────────────────────────────────────────────────────────

def _assign_speakers(whisper_result: dict, diarization: Any) -> list[dict]:
    """Merge Whisper segments with pyannote speaker turns via midpoint lookup.

    community-1 returns DiarizeOutput; older models return Annotation directly.
    We use exclusive_speaker_diarization (overlaps removed) so every moment
    maps to exactly one speaker — correct for subtitle-style assignment.
    """
    from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput

    if isinstance(diarization, DiarizeOutput):
        annotation = diarization.exclusive_speaker_diarization
    else:
        annotation = diarization  # legacy Annotation object

    # Build a flat list of (start, end, speaker) for O(n) midpoint lookup
    turns = [
        (turn.start, turn.end, label)
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]

    segments = []
    for seg in whisper_result.get("segments", []):
        start, end = seg["start"], seg["end"]
        mid = (start + end) / 2.0

        speaker = "UNKNOWN"
        for t_start, t_end, label in turns:
            if t_start <= mid <= t_end:
                speaker = label
                break

        segments.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker": speaker,
            "hu_text": seg["text"].strip(),
        })
    return segments


# ── Public entry points ───────────────────────────────────────────────────────

def run(wav_path: Path, whisper_checkpoint: Path, hf_token: str,
        language: str = "hu") -> list[dict]:
    """Full STT + diarization. Saves Whisper checkpoint before diarization runs."""
    whisper_result = _transcribe(wav_path, language)
    save_whisper_checkpoint(whisper_result, whisper_checkpoint)
    print(f"[transcribe] Whisper checkpoint saved: {whisper_checkpoint}")
    return diarize_only(wav_path, whisper_checkpoint, hf_token)


def diarize_only(wav_path: Path, whisper_checkpoint: Path,
                 hf_token: str) -> list[dict]:
    """Run only pyannote against an existing Whisper checkpoint."""
    if not whisper_checkpoint.exists():
        raise FileNotFoundError(
            f"Whisper checkpoint not found: {whisper_checkpoint}\n"
            "Run without --skip-whisper first."
        )
    whisper_result = load_whisper_checkpoint(whisper_checkpoint)
    diarization = _diarize(wav_path, hf_token)
    segments = _assign_speakers(whisper_result, diarization)
    print(f"[transcribe] {len(segments)} segments, "
          f"{len(set(s['speaker'] for s in segments))} speakers")
    return segments
