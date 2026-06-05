from __future__ import annotations

import argparse
import json
import re
import socket
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cache import TranslationCache
from translator import (
    OllamaTranslationError,
    OllamaTranslator,
    PROMPT_VERSION,
    REFINE_PROMPT_VERSION,
)


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
DRAFTS_DIR = ROOT / "drafts"
CACHE_PATH = ROOT / ".cache" / "translation_cache.json"
GLOSSARY_PATH = ROOT / "glossary.json"


translator = OllamaTranslator(glossary_path=GLOSSARY_PATH)
cache = TranslationCache(CACHE_PATH)
api = FastAPI(title="Interactive Bilingual Writer")
api.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class Segment(BaseModel):
    id: str
    text: str
    previous_text: Optional[str] = None
    next_text: Optional[str] = None
    previous_translation: Optional[str] = None


class TranslateRequest(BaseModel):
    direction: str = "auto"
    mode: str = "quick"
    source_text: str = ""
    export_format: str = "sections"
    segments: List[Segment] = Field(default_factory=list)


class SaveRequest(BaseModel):
    source: str = ""
    translation: str = ""
    source_lang: str = "Source"
    target_lang: str = "Translation"
    export_format: str = "sections"


@api.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@api.get("/index.html")
def index_html() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@api.post("/api/translate")
def translate(payload: TranslateRequest) -> JSONResponse:
    try:
        mode = payload.mode if payload.mode in ("quick", "refine") else "quick"
        export_format = normalize_export_format(payload.export_format)
        combined_text = "\n\n".join(segment.text for segment in payload.segments)
        source_lang, target_lang = translator.normalize_direction(
            payload.direction,
            combined_text,
        )
        glossary_hash = translator.glossary_hash()
        results: List[Dict[str, Any]] = []
        stats = {"cached": 0, "translated": 0, "skipped": 0, "quick_reused": 0, "warnings": 0}
        all_warnings: List[Dict[str, Any]] = []

        for index_value, segment in enumerate(payload.segments):
            segment_id = segment.id or str(index_value)
            text = segment.text
            refine_context = (
                context_for_refine(payload.segments, index_value, results)
                if mode == "refine"
                else None
            )
            context_hash = (
                refine_context_hash(refine_context)
                if refine_context is not None
                else None
            )

            if translator.should_skip_translation(text):
                warnings = structure_warnings(text, text)
                results.append(
                    {
                        "id": segment_id,
                        "translation": text,
                        "cached": False,
                        "skipped": True,
                        "warnings": warnings,
                    }
                )
                stats["skipped"] += 1
                continue

            key = make_cache_key(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                mode=mode,
                glossary_hash=glossary_hash,
                context_hash=context_hash,
            )
            cached_value = cache.get(key)
            if cached_value is not None:
                translation = cached_value
                cached = True
                stats["cached"] += 1
            else:
                initial_translation = None
                if mode == "refine":
                    quick_key = make_cache_key(
                        text=text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        mode="quick",
                        glossary_hash=glossary_hash,
                    )
                    initial_translation = cache.get(quick_key)
                    if initial_translation is None:
                        initial_translation = translator.translate(
                            text,
                            source_lang,
                            target_lang,
                            "quick",
                        )
                        cache.set(
                            quick_key,
                            initial_translation,
                            cache_metadata(
                                source_lang,
                                target_lang,
                                "quick",
                                glossary_hash,
                            ),
                        )
                    else:
                        stats["quick_reused"] += 1

                if mode == "refine":
                    translation = translator.refine(
                        text,
                        source_lang,
                        target_lang,
                        initial_translation or "",
                        context=refine_context,
                    )
                else:
                    translation = translator.translate(text, source_lang, target_lang, mode)
                cache.set(
                    key,
                    translation,
                    cache_metadata(
                        source_lang,
                        target_lang,
                        mode,
                        glossary_hash,
                        context_hash=context_hash,
                    ),
                )
                cached = False
                stats["translated"] += 1

            warnings = structure_warnings(text, translation)
            if warnings:
                stats["warnings"] += len(warnings)
                all_warnings.append({"id": segment_id, "warnings": warnings})

            results.append(
                {
                    "id": segment_id,
                    "translation": translation,
                    "cached": cached,
                    "skipped": False,
                    "warnings": warnings,
                }
            )

        translation_text = "\n\n".join(item["translation"] for item in results)
        if payload.source_text.strip() or translation_text.strip():
            save_latest_draft(
                payload.source_text,
                translation_text,
                source_lang,
                target_lang,
                export_format,
            )

        return JSONResponse(
            {
                "source_lang": source_lang,
                "target_lang": target_lang,
                "model": translator.model,
                "mode": mode,
                "segments": results,
                "stats": stats,
                "structure_warnings": all_warnings,
            }
        )
    except OllamaTranslationError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    except Exception as exc:  # noqa: BLE001 - local tool should report failure plainly.
        return JSONResponse({"error": str(exc)}, status_code=500)


@api.post("/api/translate/stream")
def translate_stream(payload: TranslateRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_translation_events(payload),
        media_type="application/x-ndjson",
    )


@api.post("/api/save")
def save(payload: SaveRequest) -> JSONResponse:
    path = save_snapshot(
        payload.source,
        payload.translation,
        payload.source_lang,
        payload.target_lang,
        normalize_export_format(payload.export_format),
    )
    return JSONResponse({"path": str(path)})


@api.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse(translator.health_check())


def stream_translation_events(payload: TranslateRequest) -> Iterator[str]:
    try:
        mode = payload.mode if payload.mode in ("quick", "refine") else "quick"
        export_format = normalize_export_format(payload.export_format)
        combined_text = "\n\n".join(segment.text for segment in payload.segments)
        source_lang, target_lang = translator.normalize_direction(
            payload.direction,
            combined_text,
        )
        glossary_hash = translator.glossary_hash()
        results: List[Dict[str, Any]] = []
        stats = {"cached": 0, "translated": 0, "skipped": 0, "quick_reused": 0, "warnings": 0}
        all_warnings: List[Dict[str, Any]] = []

        yield ndjson_event(
            {
                "type": "meta",
                "source_lang": source_lang,
                "target_lang": target_lang,
                "model": translator.model,
                "mode": mode,
            }
        )

        for index_value, segment in enumerate(payload.segments):
            segment_id = segment.id or str(index_value)
            text = segment.text
            refine_context = (
                context_for_refine(payload.segments, index_value, results)
                if mode == "refine"
                else None
            )
            context_hash = (
                refine_context_hash(refine_context)
                if refine_context is not None
                else None
            )

            if translator.should_skip_translation(text):
                warnings = structure_warnings(text, text)
                result = {
                    "id": segment_id,
                    "translation": text,
                    "cached": False,
                    "skipped": True,
                    "warnings": warnings,
                }
                results.append(result)
                stats["skipped"] += 1
                yield ndjson_event({"type": "segment_done", **result})
                continue

            key = make_cache_key(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                mode=mode,
                glossary_hash=glossary_hash,
                context_hash=context_hash,
            )
            cached_value = cache.get(key)
            if cached_value is not None:
                warnings = structure_warnings(text, cached_value)
                if warnings:
                    stats["warnings"] += len(warnings)
                    all_warnings.append({"id": segment_id, "warnings": warnings})
                result = {
                    "id": segment_id,
                    "translation": cached_value,
                    "cached": True,
                    "skipped": False,
                    "warnings": warnings,
                }
                results.append(result)
                stats["cached"] += 1
                yield ndjson_event({"type": "segment_done", **result})
                continue

            initial_translation = None
            if mode == "refine":
                quick_key = make_cache_key(
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    mode="quick",
                    glossary_hash=glossary_hash,
                )
                initial_translation = cache.get(quick_key)
                if initial_translation is None:
                    yield ndjson_event({"type": "segment_status", "id": segment_id, "status": "quick"})
                    initial_translation = translator.translate(text, source_lang, target_lang, "quick")
                    cache.set(
                        quick_key,
                        initial_translation,
                        cache_metadata(source_lang, target_lang, "quick", glossary_hash),
                    )
                else:
                    stats["quick_reused"] += 1
                yield ndjson_event({"type": "segment_status", "id": segment_id, "status": "reflecting"})

            yield ndjson_event({"type": "segment_start", "id": segment_id})
            chunks: List[str] = []
            for chunk in translator.translate_stream(
                text,
                source_lang,
                target_lang,
                mode,
                initial=initial_translation,
                context=refine_context,
            ):
                chunks.append(chunk)
                yield ndjson_event({"type": "chunk", "id": segment_id, "text": chunk})

            translation = translator._strip_thinking("".join(chunks))
            warnings = structure_warnings(text, translation)
            if warnings:
                stats["warnings"] += len(warnings)
                all_warnings.append({"id": segment_id, "warnings": warnings})
            cache.set(
                key,
                translation,
                cache_metadata(
                    source_lang,
                    target_lang,
                    mode,
                    glossary_hash,
                    context_hash=context_hash,
                ),
            )
            stats["translated"] += 1
            result = {
                "id": segment_id,
                "translation": translation,
                "cached": False,
                "skipped": False,
                "warnings": warnings,
            }
            results.append(result)
            yield ndjson_event({"type": "segment_done", **result})

        translation_text = "\n\n".join(item["translation"] for item in results)
        if payload.source_text.strip() or translation_text.strip():
            save_latest_draft(
                payload.source_text,
                translation_text,
                source_lang,
                target_lang,
                export_format,
            )

        yield ndjson_event(
            {
                "type": "done",
                "stats": stats,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "model": translator.model,
                "mode": mode,
                "structure_warnings": all_warnings,
            }
        )
    except OllamaTranslationError as exc:
        yield ndjson_event({"type": "error", "message": str(exc)})
    except Exception as exc:  # noqa: BLE001 - local tool should report failure plainly.
        yield ndjson_event({"type": "error", "message": str(exc)})


def ndjson_event(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def markdown_structure_counts(text: str) -> Dict[str, int]:
    return {
        "headings": len(re.findall(r"(?m)^\s{0,3}#{1,6}\s+", text)),
        "code blocks": text.count("```") // 2,
        "links": len(re.findall(r"(?<!!)\[[^\]]+\]\([^)]+\)", text)),
        "images": len(re.findall(r"!\[[^\]]*\]\([^)]+\)", text)),
        "tables": len(
            re.findall(
                r"(?m)^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$",
                text,
            )
        ),
        "list items": len(re.findall(r"(?m)^\s*(?:[-*+]\s+|\d+\.\s+)", text)),
    }


def structure_warnings(source: str, translation: str) -> List[str]:
    source_counts = markdown_structure_counts(source)
    translation_counts = markdown_structure_counts(translation)
    warnings = []
    for label, source_count in source_counts.items():
        target_count = translation_counts[label]
        if source_count != target_count:
            warnings.append(f"{label} {source_count}->{target_count}")
    return warnings


def make_cache_key(
    *,
    text: str,
    source_lang: str,
    target_lang: str,
    mode: str,
    glossary_hash: str,
    context_hash: Optional[str] = None,
) -> str:
    return cache.make_key(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        model=translator.model,
        mode=mode,
        prompt_version=prompt_version_for_mode(mode),
        glossary_hash=glossary_hash,
        context_hash=context_hash,
    )


def cache_metadata(
    source_lang: str,
    target_lang: str,
    mode: str,
    glossary_hash: str,
    context_hash: Optional[str] = None,
) -> Dict[str, str]:
    metadata = {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "model": translator.model,
        "mode": mode,
        "prompt_version": prompt_version_for_mode(mode),
        "glossary_hash": glossary_hash,
    }
    if context_hash is not None:
        metadata["context_hash"] = context_hash
    return metadata


def prompt_version_for_mode(mode: str) -> str:
    return REFINE_PROMPT_VERSION if mode == "refine" else PROMPT_VERSION


def context_for_refine(
    segments: List[Segment],
    index_value: int,
    results: List[Dict[str, Any]],
) -> Dict[str, str]:
    segment = segments[index_value]
    previous_source = (
        segment.previous_text
        if segment.previous_text is not None
        else (segments[index_value - 1].text if index_value > 0 else "")
    )
    next_source = (
        segment.next_text
        if segment.next_text is not None
        else (segments[index_value + 1].text if index_value + 1 < len(segments) else "")
    )

    previous_translation = ""
    if (
        index_value > 0
        and previous_source == segments[index_value - 1].text
        and results
    ):
        previous_translation = str(results[-1].get("translation") or "")
    elif segment.previous_translation:
        previous_translation = segment.previous_translation

    return {
        "previous_source": previous_source or "",
        "previous_translation": previous_translation or "",
        "next_source": next_source or "",
    }


def refine_context_hash(context: Dict[str, str]) -> str:
    encoded = json.dumps(context, ensure_ascii=False, sort_keys=True)
    return TranslationCache.hash_text(encoded)


def save_latest_draft(
    source: str,
    translation: str,
    source_lang: str,
    target_lang: str,
    export_format: str,
) -> None:
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    latest = DRAFTS_DIR / "latest.md"
    latest.write_text(
        format_draft(source, translation, source_lang, target_lang, export_format),
        encoding="utf-8",
    )


def save_snapshot(
    source: str,
    translation: str,
    source_lang: str,
    target_lang: str,
    export_format: str,
) -> Path:
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = DRAFTS_DIR / f"draft-{stamp}.md"
    path.write_text(
        format_draft(source, translation, source_lang, target_lang, export_format),
        encoding="utf-8",
    )
    return path


def format_draft(
    source: str,
    translation: str,
    source_lang: str,
    target_lang: str,
    export_format: str,
) -> str:
    source = source.strip()
    translation = translation.strip()
    if export_format == "target":
        return f"{translation}\n"
    if export_format == "interleaved":
        return format_interleaved(source, translation, source_lang, target_lang)
    if export_format == "table":
        return format_table(source, translation, source_lang, target_lang)
    return f"# Bilingual Draft\n\n## {source_lang}\n\n{source}\n\n## {target_lang}\n\n{translation}\n"


def normalize_export_format(export_format: str) -> str:
    if export_format in {"sections", "target", "interleaved", "table"}:
        return export_format
    return "sections"


def split_paragraphs(text: str) -> List[str]:
    return [part.strip() for part in text.split("\n\n") if part.strip()]


def format_interleaved(
    source: str,
    translation: str,
    source_lang: str,
    target_lang: str,
) -> str:
    source_parts = split_paragraphs(source)
    target_parts = split_paragraphs(translation)
    lines = ["# Bilingual Draft", ""]
    max_len = max(len(source_parts), len(target_parts))
    for index_value in range(max_len):
        if index_value < len(source_parts):
            lines.extend([f"**{source_lang}**", "", source_parts[index_value], ""])
        if index_value < len(target_parts):
            lines.extend([f"**{target_lang}**", "", target_parts[index_value], ""])
    return "\n".join(lines).rstrip() + "\n"


def format_table(
    source: str,
    translation: str,
    source_lang: str,
    target_lang: str,
) -> str:
    source_parts = split_paragraphs(source)
    target_parts = split_paragraphs(translation)
    lines = [
        f"| {source_lang} | {target_lang} |",
        "| --- | --- |",
    ]
    max_len = max(len(source_parts), len(target_parts))
    for index_value in range(max_len):
        source_text = source_parts[index_value] if index_value < len(source_parts) else ""
        target_text = target_parts[index_value] if index_value < len(target_parts) else ""
        lines.append(f"| {escape_table_cell(source_text)} | {escape_table_cell(target_text)} |")
    return "\n".join(lines) + "\n"


def escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Local bilingual writing editor.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()

    if not port_is_available(args.host, args.port):
        print(f"Port {args.port} is already in use.")
        print(f"Stop the existing server or run: python3 app.py --port {args.port + 1}")
        return

    try:
        import uvicorn
    except ImportError:
        print("FastAPI server dependencies are missing.")
        print("Install them with: python3 -m pip install -r requirements.txt")
        return

    print(f"Interactive writer running at http://{args.host}:{args.port}")
    print(f"Ollama endpoint: {translator.base_url}")
    print(f"Model: {translator.model}")
    uvicorn.run(api, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
