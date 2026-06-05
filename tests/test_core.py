from pathlib import Path

import json

from app import (
    Segment,
    context_for_refine,
    format_draft,
    markdown_structure_counts,
    ndjson_event,
    normalize_export_format,
    refine_context_hash,
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


def test_refine_prompt_includes_neighbor_context():
    translator = OllamaTranslator()
    context = {
        "previous_source": "The model stores facts in memory.",
        "previous_translation": "模型将事实存储在记忆中。",
        "next_source": "This makes later retrieval faster.",
    }

    prompt = translator._improvement_prompt(
        "It then answers the question.",
        "English",
        "Chinese",
        "然后它回答问题。",
        "Keep pronouns consistent with the previous paragraph.",
        context=context,
    )

    assert "Previous English paragraph:" in prompt
    assert "Previous Chinese translation:" in prompt
    assert "Next English paragraph:" in prompt
    assert "模型将事实存储在记忆中。" in prompt
    assert "Output only the improved Chinese translation of the current paragraph." in prompt


def test_context_for_refine_prefers_current_run_previous_translation():
    segments = [
        Segment(id="0", text="Previous source."),
        Segment(
            id="1",
            text="Current source.",
            previous_text="Previous source.",
            previous_translation="Stale previous translation.",
            next_text="Next source.",
        ),
    ]
    results = [{"id": "0", "translation": "Fresh previous translation."}]

    context = context_for_refine(segments, 1, results)

    assert context["previous_source"] == "Previous source."
    assert context["previous_translation"] == "Fresh previous translation."
    assert context["next_source"] == "Next source."


def test_context_for_refine_uses_preview_translation_when_previous_is_omitted():
    segments = [
        Segment(id="0", text="Current source."),
        Segment(
            id="2",
            text="Later source.",
            previous_text="Omitted active source.",
            previous_translation="Preview translation.",
            next_text="Next source.",
        ),
    ]
    results = [{"id": "0", "translation": "Current translation."}]

    context = context_for_refine(segments, 1, results)

    assert context["previous_source"] == "Omitted active source."
    assert context["previous_translation"] == "Preview translation."
    assert context["next_source"] == "Next source."


def test_context_hash_changes_cache_key():
    base = TranslationCache.make_key(
        text="Current source.",
        source_lang="English",
        target_lang="Chinese",
        model="test-model",
        mode="refine",
        prompt_version="refine-context-v1",
        glossary_hash="glossary",
        context_hash=refine_context_hash(
            {
                "previous_source": "A",
                "previous_translation": "甲",
                "next_source": "B",
            }
        ),
    )
    changed = TranslationCache.make_key(
        text="Current source.",
        source_lang="English",
        target_lang="Chinese",
        model="test-model",
        mode="refine",
        prompt_version="refine-context-v1",
        glossary_hash="glossary",
        context_hash=refine_context_hash(
            {
                "previous_source": "A",
                "previous_translation": "甲",
                "next_source": "Changed next source.",
            }
        ),
    )

    assert base != changed


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
