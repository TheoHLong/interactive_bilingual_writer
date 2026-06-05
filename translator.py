from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple


# v3: glossary entries are directional and reversed for Chinese-to-English.
PROMPT_VERSION = "quick-markdown-v3"
REFINE_PROMPT_VERSION = "refine-context-v1"


class OllamaTranslationError(RuntimeError):
    pass


class OllamaTranslator:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 300,
        glossary_path: Optional[Path] = None,
    ):
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434/api/generate"
        )
        self.model = model or os.getenv("OLLAMA_TRANSLATION_MODEL", "translategemma")
        self.timeout = timeout
        self.glossary_path = glossary_path

    def load_glossary(self) -> Dict[str, str]:
        if not self.glossary_path or not self.glossary_path.exists():
            return {}

        try:
            with self.glossary_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

        if not isinstance(data, dict):
            return {}

        normalized: Dict[str, str] = {}
        for key, value in data.items():
            if not str(key).strip():
                continue
            if isinstance(value, dict):
                english = value.get("en") or value.get("english") or key
                chinese = value.get("zh") or value.get("chinese")
                if chinese:
                    normalized[str(english)] = str(chinese)
            else:
                normalized[str(key)] = str(value)
        return normalized

    def glossary_hash(self) -> str:
        glossary = self.load_glossary()
        encoded = json.dumps(glossary, ensure_ascii=False, sort_keys=True)
        import hashlib

        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def detect_language(text: str) -> str:
        compact = re.sub(r"\s+", "", text)
        if not compact:
            return "English"

        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", compact))
        latin_count = len(re.findall(r"[A-Za-z]", compact))
        total_signal = cjk_count + latin_count
        if total_signal == 0:
            return "English"

        return "Chinese" if cjk_count / total_signal >= 0.2 else "English"

    @staticmethod
    def target_for(source_lang: str) -> str:
        return "English" if source_lang == "Chinese" else "Chinese"

    @staticmethod
    def normalize_direction(direction: str, combined_text: str) -> Tuple[str, str]:
        if direction == "zh-en":
            return "Chinese", "English"
        if direction == "en-zh":
            return "English", "Chinese"

        source_lang = OllamaTranslator.detect_language(combined_text)
        return source_lang, OllamaTranslator.target_for(source_lang)

    @staticmethod
    def should_skip_translation(text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        if stripped.startswith("```") and stripped.endswith("```"):
            return True
        if re.fullmatch(r"\$\$[\s\S]*\$\$", stripped):
            return True
        return False

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        mode: str,
        context: Optional[Dict[str, str]] = None,
    ) -> str:
        if mode == "refine":
            initial = self._translate_quick(text, source_lang, target_lang)
            return self.refine(text, source_lang, target_lang, initial, context=context)
        return self._translate_quick(text, source_lang, target_lang)

    def translate_stream(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        mode: str,
        initial: Optional[str] = None,
        context: Optional[Dict[str, str]] = None,
    ) -> Iterator[str]:
        if mode == "refine":
            quick_initial = initial or self._translate_quick(text, source_lang, target_lang)
            yield from self.refine_stream(
                text,
                source_lang,
                target_lang,
                quick_initial,
                context=context,
            )
            return

        prompt, system_message = self._quick_prompt(text, source_lang, target_lang)
        yield from self._completion_stream(prompt, system_message)

    def refine(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        quick_initial: str,
        context: Optional[Dict[str, str]] = None,
    ) -> str:
        return self._translate_refined(
            text,
            source_lang,
            target_lang,
            quick_initial,
            context=context,
        )

    def refine_stream(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        quick_initial: str,
        context: Optional[Dict[str, str]] = None,
    ) -> Iterator[str]:
        notes = self._reflection_notes(
            text,
            source_lang,
            target_lang,
            quick_initial,
            context=context,
        )
        final_prompt = self._improvement_prompt(
            text,
            source_lang,
            target_lang,
            quick_initial,
            notes,
            context=context,
        )
        yield from self._completion_stream(
            final_prompt,
            "You are an expert translation editor.",
        )

    def _translate_quick(self, text: str, source_lang: str, target_lang: str) -> str:
        prompt, system_message = self._quick_prompt(text, source_lang, target_lang)
        return self._completion(prompt, system_message)

    def _quick_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> Tuple[str, str]:
        glossary = self.load_glossary()
        glossary_block = self._format_glossary(glossary, source_lang, target_lang)
        system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."
        prompt = f"""This is a {source_lang} to {target_lang} translation. Please provide the {target_lang} translation for this text.
Do not provide any explanations or text apart from the translation.
Preserve Markdown syntax, headings, lists, links, code fences, inline code, URLs, and LaTeX math.
Translate natural-language prose only.
{glossary_block}

{source_lang}: {text}

{target_lang}:
"""
        return prompt, system_message

    def _translate_refined(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        initial: str,
        context: Optional[Dict[str, str]] = None,
    ) -> str:
        notes = self._reflection_notes(
            text,
            source_lang,
            target_lang,
            initial,
            context=context,
        )
        final_prompt = self._improvement_prompt(
            text,
            source_lang,
            target_lang,
            initial,
            notes,
            context=context,
        )
        return self._completion(final_prompt, "You are an expert translation editor.")

    def _reflection_notes(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        initial: str,
        context: Optional[Dict[str, str]] = None,
    ) -> str:
        glossary = self.load_glossary()
        glossary_block = self._format_glossary(glossary, source_lang, target_lang)
        context_block = self._format_refine_context(context, source_lang, target_lang)
        reflection_prompt = f"""Review this {source_lang} to {target_lang} translation of the current paragraph.
Use the neighboring context only to resolve continuity, pronouns, terminology, and tone.
Do not translate the neighboring paragraphs.
Preserve Markdown syntax, headings, lists, links, code fences, inline code, URLs, and LaTeX math.
{glossary_block}

{context_block}

Current source paragraph:
{text}

Current translation:
{initial}

Give concise, specific improvement notes for the current paragraph only.
"""
        notes = self._completion(
            reflection_prompt,
            "You are an expert translation reviewer.",
        )
        return notes

    def _improvement_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        initial: str,
        notes: str,
        context: Optional[Dict[str, str]] = None,
    ) -> str:
        glossary = self.load_glossary()
        glossary_block = self._format_glossary(glossary, source_lang, target_lang)
        context_block = self._format_refine_context(context, source_lang, target_lang)
        final_prompt = f"""Improve only the current paragraph translation using the notes and neighboring context.
Use context for continuity, pronouns, terminology, and tone, but do not translate neighboring paragraphs.
Preserve Markdown syntax, headings, lists, links, code fences, inline code, URLs, and LaTeX math.
{glossary_block}

{context_block}

Current source paragraph:
{text}

Current translation:
{initial}

Notes:
{notes}

Output only the improved {target_lang} translation of the current paragraph.
"""
        return final_prompt

    @staticmethod
    def _format_refine_context(
        context: Optional[Dict[str, str]],
        source_lang: str,
        target_lang: str,
    ) -> str:
        context = context or {}
        previous_source = (context.get("previous_source") or "").strip()
        previous_translation = (context.get("previous_translation") or "").strip()
        next_source = (context.get("next_source") or "").strip()

        lines = [
            "Neighboring context:",
            "(Use this for continuity only; translate only the current paragraph.)",
        ]
        if previous_source:
            lines.extend([f"Previous {source_lang} paragraph:", previous_source])
        if previous_translation:
            lines.extend([f"Previous {target_lang} translation:", previous_translation])
        if next_source:
            lines.extend([f"Next {source_lang} paragraph:", next_source])
        if len(lines) == 2:
            lines.append("(none)")
        return "\n".join(lines)

    @staticmethod
    def _format_glossary(
        glossary: Dict[str, str],
        source_lang: str,
        target_lang: str,
    ) -> str:
        if not glossary:
            return ""

        if source_lang == "Chinese" and target_lang == "English":
            terms = [(target, source) for source, target in glossary.items()]
        else:
            terms = list(glossary.items())

        lines = ["- Follow this glossary exactly when relevant:"]
        for source, target in sorted(terms):
            if not source or not target:
                continue
            lines.append(f"  - {source}: {target}")
        return "\n".join(lines)

    def health_check(self, timeout: float = 2.0) -> Dict[str, Any]:
        tags_url = self._tags_url()
        try:
            with urllib.request.urlopen(tags_url, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
        except Exception as exc:  # noqa: BLE001 - local status should be plain.
            return {
                "ok": False,
                "online": False,
                "model": self.model,
                "model_available": False,
                "error": str(exc),
            }

        model_names = [
            str(item.get("name", ""))
            for item in data.get("models", [])
            if isinstance(item, dict)
        ]
        model_available = any(
            name == self.model or name.split(":", 1)[0] == self.model
            for name in model_names
        )
        return {
            "ok": model_available,
            "online": True,
            "model": self.model,
            "model_available": model_available,
            "models": model_names,
            "error": "" if model_available else f"Model {self.model} is not installed.",
        }

    def _tags_url(self) -> str:
        parsed = urllib.parse.urlparse(self.base_url)
        return urllib.parse.urlunparse(
            parsed._replace(path="/api/tags", params="", query="", fragment="")
        )

    def _completion(self, prompt: str, system_message: str) -> str:
        raw = self._post_generate(prompt, system_message, stream=False)

        try:
            data: Dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaTranslationError("Ollama returned invalid JSON.") from exc

        result = str(data.get("response", "")).strip()
        return self._strip_thinking(result)

    def _completion_stream(self, prompt: str, system_message: str) -> Iterator[str]:
        yield from self._hide_thinking_chunks(
            self._raw_completion_stream(prompt, system_message)
        )

    def _raw_completion_stream(self, prompt: str, system_message: str) -> Iterator[str]:
        request = self._generate_request(prompt, system_message, stream=True)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise OllamaTranslationError(
                            "Ollama returned invalid streaming JSON."
                        ) from exc
                    chunk = str(data.get("response", ""))
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break
        except urllib.error.URLError as exc:
            raise OllamaTranslationError(
                f"Cannot reach Ollama at {self.base_url}: {exc}"
            ) from exc
        except TimeoutError as exc:
            raise OllamaTranslationError("Ollama request timed out.") from exc

    @classmethod
    def _hide_thinking_chunks(cls, chunks: Iterator[str]) -> Iterator[str]:
        in_think = False
        carry = ""
        for chunk in chunks:
            text = carry + chunk
            carry = ""
            while text:
                tag = "</think>" if in_think else "<think>"
                index = text.find(tag)
                if index >= 0:
                    if in_think:
                        text = text[index + len(tag) :]
                        in_think = False
                    else:
                        visible = text[:index]
                        if visible:
                            yield visible
                        text = text[index + len(tag) :]
                        in_think = True
                    continue

                if in_think:
                    carry = cls._partial_tag_suffix(text, "</think>")
                else:
                    carry = cls._partial_tag_suffix(text, "<think>")
                    visible = text[: len(text) - len(carry)] if carry else text
                    if visible:
                        yield visible
                break

        if carry and not in_think:
            yield carry

    @staticmethod
    def _partial_tag_suffix(text: str, tag: str) -> str:
        max_length = min(len(tag) - 1, len(text))
        for length in range(max_length, 0, -1):
            suffix = text[-length:]
            if tag.startswith(suffix):
                return suffix
        return ""

    def _post_generate(self, prompt: str, system_message: str, stream: bool) -> str:
        request = self._generate_request(prompt, system_message, stream)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise OllamaTranslationError(
                f"Cannot reach Ollama at {self.base_url}: {exc}"
            ) from exc
        except TimeoutError as exc:
            raise OllamaTranslationError("Ollama request timed out.") from exc

    def _generate_request(
        self,
        prompt: str,
        system_message: str,
        stream: bool,
    ) -> urllib.request.Request:
        payload = {
            "model": self.model,
            "prompt": f"{system_message}\n\n{prompt}",
            "stream": stream,
        }
        return urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    @staticmethod
    def _strip_thinking(text: str) -> str:
        if "</think>" in text:
            return text.split("</think>")[-1].strip()
        return text.strip()
