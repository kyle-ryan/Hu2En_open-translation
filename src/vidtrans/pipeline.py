"""CLI entrypoint — ties together audio extraction, STT, diarization, and translation."""

import argparse
import os
from pathlib import Path

from vidtrans import audio, transcribe, translate, formats


def _resolve_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "Set HF_TOKEN env var to your HuggingFace token.\n"
            "pyannote/speaker-diarization-community-1 is a gated model — "
            "accept terms at https://huggingface.co/pyannote/speaker-diarization-community-1 first."
        )
    return token


def main() -> None:
    parser = argparse.ArgumentParser(
        description="vidtrans — local AV1 video → speaker-labelled English SRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline stages (all local, no cloud):
  1. ffmpeg      — extract 16 kHz mono WAV
  2. mlx-whisper — speech-to-text  →  <stem>.whisper.json
  3. pyannote    — speaker diarization  ─┐
                                         ├→  <stem>.json
  4. Qwen3-32B   — HU → EN translation  ─┘
  5. formats     — write <stem>.en.srt (and optionally <stem>.hu.srt)
""",
    )
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output directory (default: same directory as input video)"
    )
    parser.add_argument(
        "--language", default="hu",
        help="Whisper source language code (default: hu)"
    )
    parser.add_argument(
        "--domain",
        default="parliamentary session recording",
        help='One-line description of the content type passed to the translation model '
             '(default: "parliamentary session recording"). '
             'Example: "political documentary", "medical lecture", "business meeting"'
    )
    parser.add_argument(
        "--hu-srt", action="store_true",
        help="Also write a Hungarian SRT alongside the English one"
    )

    resume = parser.add_argument_group(
        "resume flags",
        "Skip completed stages and resume from a checkpoint"
    )
    resume.add_argument(
        "--skip-whisper", action="store_true",
        help="Whisper already ran — load <stem>.whisper.json and run diarization only"
    )
    resume.add_argument(
        "--skip-transcribe", action="store_true",
        help="Both Whisper and diarization done — load <stem>.json, go straight to translation"
    )
    resume.add_argument(
        "--skip-translate", action="store_true",
        help="Skip translation and write SRT from source-language text only"
    )

    args = parser.parse_args()

    if args.skip_transcribe and args.skip_whisper:
        raise SystemExit("--skip-transcribe already implies Whisper is done; "
                         "don't combine with --skip-whisper")

    video_path: Path = args.video.resolve()
    if not video_path.exists():
        raise SystemExit(f"File not found: {video_path}")

    out_dir: Path = (args.out or video_path.parent).resolve()
    stem = video_path.stem

    whisper_json  = out_dir / f"{stem}.whisper.json"
    segments_json = out_dir / f"{stem}.json"
    en_srt_path   = out_dir / f"{stem}.en.srt"
    hu_srt_path   = out_dir / f"{stem}.hu.srt"
    wav_path      = out_dir / "tmp" / f"{stem}.wav"

    # ── Stage 1: STT + diarization ───────────────────────────────────────────
    if args.skip_transcribe:
        if not segments_json.exists():
            raise SystemExit(f"--skip-transcribe set but {segments_json} not found")
        print(f"[pipeline] Loading existing segments: {segments_json}")
        segments = formats.load_json(segments_json)

    elif args.skip_whisper:
        if not whisper_json.exists():
            raise SystemExit(f"--skip-whisper set but {whisper_json} not found")
        if not wav_path.exists():
            raise SystemExit(
                f"--skip-whisper needs the extracted WAV for pyannote but "
                f"{wav_path} not found.\n"
                f"Re-extract with: ffmpeg -i {video_path} -vn -ar 16000 -ac 1 "
                f"-c:a pcm_s16le {wav_path}"
            )
        hf_token = _resolve_hf_token()
        print(f"[pipeline] Resuming from Whisper checkpoint: {whisper_json}")
        segments = transcribe.diarize_only(wav_path, whisper_json, hf_token)
        formats.save_json(segments, segments_json)
        print(f"[pipeline] Segments saved: {segments_json}")

    else:
        hf_token = _resolve_hf_token()
        wav_path = audio.extract_audio(video_path, out_dir / "tmp")
        segments = transcribe.run(wav_path, whisper_json, hf_token, language=args.language)
        formats.save_json(segments, segments_json)
        print(f"[pipeline] Segments saved: {segments_json}")

    # ── Stage 2: translation ─────────────────────────────────────────────────
    if not args.skip_translate:
        segments = translate.run(segments, domain_hint=args.domain)
        formats.save_json(segments, segments_json)
        print(f"[pipeline] Translations saved: {segments_json}")

    # ── Stage 3: SRT output ──────────────────────────────────────────────────
    text_field = "hu_text" if args.skip_translate else "en_text"
    formats.save_srt(segments, en_srt_path, text_field=text_field)
    print(f"[pipeline] SRT written: {en_srt_path}")

    if args.hu_srt:
        formats.save_srt(segments, hu_srt_path, text_field="hu_text")
        print(f"[pipeline] Hungarian SRT written: {hu_srt_path}")
