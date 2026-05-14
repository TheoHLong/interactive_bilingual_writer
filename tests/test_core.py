from pathlib import Path

import json

from app import (
    format_draft,
    markdown_structure_counts,
    ndjson_event,
    normalize_export_format,
    structure_warnings,
)
from cache import TranslationCache
from translator import OllamaTranslator


def test_detect_language():
    assert OllamaTranslator.detect_language("This is an English sentence.") == "English"
    assert OllamaTranslator.detect_language("这是中文句子。") == "Chinese"
    assert OllamaTranslator.detect_language("embedding 嵌入向量") == "Chinese"


def test_normalize_direction():
    assert OllamaTranslator.normalize_direction("en-zh", "anything") == (
        "English",
        "Chinese",
    )
    assert OllamaTranslator.normalize_direction("zh-en", "anything") == (
        "Chinese",
        "English",
    )
    assert OllamaTranslator.normalize_direction("auto", "这是中文。") == (
        "Chinese",
        "English",
    )


def test_should_skip_translation():
    assert OllamaTranslator.should_skip_translation("")
    assert OllamaTranslator.should_skip_translation("```python\nprint('x')\n```")
    assert OllamaTranslator.should_skip_translation("$$x + y$$")
    assert not OllamaTranslator.should_skip_translation("Translate this paragraph.")


def test_format_glossary_directional():
    glossary = {"embedding": "嵌入向量", "Transformer": "Transformer"}

    en_to_zh = OllamaTranslator._format_glossary(glossary, "English", "Chinese")
    assert "embedding: 嵌入向量" in en_to_zh
    assert "Transformer: Transformer" in en_to_zh

    zh_to_en = OllamaTranslator._format_glossary(glossary, "Chinese", "English")
    assert "嵌入向量: embedding" in zh_to_en
    assert "Transformer: Transformer" in zh_to_en


def test_strip_thinking():
    assert OllamaTranslator._strip_thinking("<think>notes</think>Translation") == "Translation"
    assert OllamaTranslator._strip_thinking(" Translation ") == "Translation"


def test_hide_thinking_chunks():
    chunks = ["<thi", "nk>hidden", "</thi", "nk>Visible", " text"]
    assert "".join(OllamaTranslator._hide_thinking_chunks(iter(chunks))) == "Visible text"


def test_generate_request_stream_flag():
    translator = OllamaTranslator(model="test-model")
    request = translator._generate_request("Prompt", "System", stream=True)
    payload = json.loads(request.data.decode("utf-8"))

    assert payload["model"] == "test-model"
    assert payload["stream"] is True
    assert payload["prompt"] == "System\n\nPrompt"


def test_ndjson_event():
    line = ndjson_event({"type": "chunk", "text": "你好"})
    assert line.endswith("\n")
    assert json.loads(line)["text"] == "你好"


def test_export_formats():
    source = "Hello.\n\nWorld."
    target = "你好。\n\n世界。"

    assert normalize_export_format("unknown") == "sections"
    assert format_draft(source, target, "English", "Chinese", "target") == "你好。\n\n世界。\n"

    interleaved = format_draft(source, target, "English", "Chinese", "interleaved")
    assert "<!--" not in interleaved
    assert "**English**" in interleaved
    assert "**Chinese**" in interleaved

    table = format_draft(source, target, "English", "Chinese", "table")
    assert "| English | Chinese |" in table
    assert "| Hello. | 你好。 |" in table


def test_structure_warnings():
    source = "# Title\n\n```python\nprint('x')\n```\n\n[link](https://example.com)"
    same = "# 标题\n\n```python\nprint('x')\n```\n\n[链接](https://example.com)"
    broken = "# 标题\n\n[链接](https://example.com)"

    assert markdown_structure_counts(source)["code blocks"] == 1
    assert structure_warnings(source, same) == []
    assert "code blocks 1->0" in structure_warnings(source, broken)


def test_cache_prunes_old_entries(tmp_path: Path):
    cache = TranslationCache(tmp_path / "cache.json", max_entries=2)
    cache.set("one", "1", {})
    cache.set("two", "2", {})
    cache.get("one")
    cache.set("three", "3", {})

    assert cache.get("one") == "1"
    assert cache.get("two") is None
    assert cache.get("three") == "3"
