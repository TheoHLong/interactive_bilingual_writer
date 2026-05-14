# Interactive Bilingual Writer

A local one-sided writing editor with a translated Markdown preview. It uses the
same Ollama generate API shape as the existing translator project:

```json
{"model": "translategemma", "prompt": "...", "stream": false}
```

The browser uses `/api/translate/stream` for live preview updates. Cached
paragraphs return immediately, uncached paragraphs stream chunks as Ollama emits
them, and the cache is written only after a paragraph finishes successfully.
Streaming output hides `<think>...</think>` blocks before they reach the browser.
Partial chunks are rendered through the same Markdown preview path as completed
paragraphs, so the live preview stays visually close to the final output.

## Run

Start Ollama and make sure the translation model is available:

```bash
ollama pull translategemma
```

Run the local server:

```bash
cd interactive_bilingual_writer
python3 -m pip install -r requirements.txt
python3 app.py
```

Open:

```text
http://127.0.0.1:8765
```

## Settings

- `Delay`: debounce time before auto translation. Default: `2` seconds.
- `Min change`: minimum changed characters before automatic translation runs. The value is used for English; Chinese text uses half of it, so the default `12` becomes `6`.
- `Active paragraph`: whether the paragraph currently being edited is translated.
- `Direction`: automatic language detection, English to Chinese, or Chinese to English.
- `Export`: manual save format: sections, target only, interleaved, or Markdown table.

The editor ignores auto-translation triggers while a Chinese/Japanese/Korean IME
composition is active, then schedules translation once composition ends.

## Files

- `app.py`: FastAPI server and API.
- `translator.py`: Ollama translation calls.
- `cache.py`: persistent paragraph translation cache with a 5000-entry LRU cap.
- `static/`: editor UI.
- `static/vendor/marked.umd.js`: local Marked renderer for Markdown preview.
- `static/vendor/highlight.min.js`: local Highlight.js renderer for code blocks.
- `glossary.json`: English-to-Chinese term mapping, automatically reversed for Chinese-to-English translation.
- `drafts/latest.md`: latest autosaved bilingual draft.
- `drafts/draft-*.md`: manual snapshots created by the Save button.
- `tests/`: pure function regression tests that do not require Ollama.

The cache key includes source text, direction, model, mode, prompt version, and
glossary hash, so prompt or glossary changes do not reuse stale translations.
Completed translations are checked for lightweight Markdown structure drift
across headings, code blocks, links, images, tables, and list items; mismatches
are reported in the preview stats tooltip.

## Test

```bash
python -m pytest -q
```
