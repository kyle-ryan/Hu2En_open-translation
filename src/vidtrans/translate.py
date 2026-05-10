"""Translate Hungarian text fields to English using local mlx-lm Qwen3-32B-8bit.

The model is loaded once, used for all chunks, then explicitly freed so the
caller can do further work without the weights sitting in the Metal cache.

Prompt design
-------------
Each chunk includes:
  - a system prompt with domain context (sanitised before injection)
  - a read-only context window of the N preceding translated lines so Qwen
    has continuity across chunk boundaries
  - speaker labels on every line so register/voice can be preserved
  - deduplication so repeated hallucinated segments are translated once and
    the result is copied, not re-inferred
"""

import gc
import re
from tqdm import tqdm

MODEL_REPO  = "mlx-community/Qwen3-32B-8bit"
CHUNK_SIZE  = 20   # segments per translation call
CONTEXT_WIN = 3    # preceding translated lines shown as read-only context

# Domain hint constraints — prevents prompt injection via --domain
_DOMAIN_MAX_LEN = 200


def _sanitise_domain(domain: str) -> str:
    """Collapse whitespace, strip control characters, cap length.

    The domain hint is injected directly into the system prompt. Without
    sanitisation a crafted value could break prompt structure or inject
    additional instructions. This is especially important now the tool is
    public and accepts arbitrary --domain values.
    """
    # Strip all control characters (includes newlines, tabs, null bytes)
    domain = re.sub(r"[\x00-\x1f\x7f]", " ", domain)
    # Collapse runs of whitespace to a single space
    domain = " ".join(domain.split())
    # Hard cap — anything longer is almost certainly not a legitimate content type
    return domain[:_DOMAIN_MAX_LEN]


def _free_mlx_memory() -> None:
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception:
        pass
    gc.collect()


def _build_prompt(
    chunk: list[dict],
    context_lines: list[str],
    domain_hint: str,          # must already be sanitised before passing in
) -> str:
    system = (
        "You are a professional Hungarian-to-English translator working on a "
        f"{domain_hint}. "
        "Translate each TRANSLATE line from Hungarian to English. "
        "Preserve each speaker's register, formality, and tone exactly. "
        "Return ONLY the numbered English translations, one per line. "
        "Do not include speaker labels, context lines, or any other text."
    )

    parts = []

    if context_lines:
        parts.append("Context (already translated, do not output these):")
        for line in context_lines:
            parts.append(f"  {line}")
        parts.append("")

    parts.append("TRANSLATE:")
    for i, seg in enumerate(chunk):
        speaker = seg.get("speaker", "UNKNOWN")
        parts.append(f"{i+1}. [{speaker}] {seg['hu_text']}")

    user_content = "\n".join(parts)
    return system + "\n\n" + user_content


def _translate_chunk(
    chunk: list[dict],
    context_lines: list[str],
    domain_hint: str,
    model,
    tokenizer,
) -> list[str]:
    from mlx_lm import generate

    prompt_text = _build_prompt(chunk, context_lines, domain_hint)
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    response = generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=CHUNK_SIZE * 150,
        verbose=False,
    )

    # Parse "N. translated text" lines; fall back to source on parse failure
    translations: dict[int, str] = {}
    for line in response.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        dot = line.find(".")
        if dot > 0 and line[:dot].isdigit():
            idx = int(line[:dot]) - 1
            translations[idx] = line[dot + 1:].strip()

    return [translations.get(i, chunk[i]["hu_text"]) for i in range(len(chunk))]


def run(segments: list[dict], domain_hint: str = "parliamentary session recording") -> list[dict]:
    from mlx_lm import load

    domain_hint = _sanitise_domain(domain_hint)

    # ── Deduplicate: build a cache of text → translation so identical
    #    segments (e.g. repeated subtitle watermarks) are inferred once.
    translation_cache: dict[str, str] = {}

    print(f"[translate] Loading {MODEL_REPO} ...")
    model, tokenizer = load(MODEL_REPO)

    result = [s.copy() for s in segments]
    context_lines: list[str] = []   # rolling window of recent translations

    chunks = [segments[i:i + CHUNK_SIZE] for i in range(0, len(segments), CHUNK_SIZE)]
    offset = 0

    try:
        for chunk in tqdm(chunks, desc="Translating", unit="chunk"):
            # Split into cache hits and misses
            needs_inference = [
                s for s in chunk if s["hu_text"] not in translation_cache
            ]

            if needs_inference:
                translations = _translate_chunk(
                    needs_inference, context_lines, domain_hint, model, tokenizer
                )
                for seg, en_text in zip(needs_inference, translations):
                    translation_cache[seg["hu_text"]] = en_text

            # Apply cached translations — use enumerate to avoid list.index()
            # which would silently write the wrong entry for duplicate hu_text
            # values within a chunk.
            chunk_translations = []
            for j, seg in enumerate(chunk):
                en_text = translation_cache[seg["hu_text"]]
                chunk_translations.append(en_text)
                result[offset + j]["en_text"] = en_text

            # Advance context window with this chunk's output
            new_ctx = [
                f"[{seg.get('speaker','?')}] {en}"
                for seg, en in zip(chunk, chunk_translations)
            ]
            context_lines = (context_lines + new_ctx)[-CONTEXT_WIN:]
            offset += len(chunk)

    finally:
        print("[translate] Freeing Qwen3 from Metal cache ...")
        del model, tokenizer
        _free_mlx_memory()

    return result
