"""Tests for code block extraction and language detection."""

import pytest
from langchoicebench.extraction.code_blocks import extract_code_blocks
from langchoicebench.extraction.languages import (
    extract_implementation_language,
    extract_suggested_languages,
    normalise_language,
)


class TestNormaliseLanguage:
    """Test the language normalisation lookup table."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("js", "javascript"),
            ("JS", "javascript"),
            ("ts", "typescript"),
            ("TypeScript", "typescript"),
            ("py", "python"),
            ("Python", "python"),
            ("python3", "python"),
            ("c++", "c++"),
            ("cpp", "c++"),
            ("c#", "c#"),
            ("csharp", "c#"),
            ("golang", "go"),
            ("flutter", "dart"),
            ("gdscript", "gdscript"),
        ],
    )
    def test_known_aliases(self, raw: str, expected: str) -> None:
        """Known aliases should normalise to lowercase canonical names."""
        assert normalise_language(raw) == expected

    def test_unknown_returns_none(self) -> None:
        """Unrecognised strings should return None."""
        assert normalise_language("brainfuck") is None
        assert normalise_language("") is None


class TestExtractSuggestedLanguages:
    """Test recommendation language extraction from model outputs."""

    def test_language_tags_extracted(self) -> None:
        """<language> tags should be the primary extraction mechanism."""
        text = "I recommend <language>Swift</language> for this project."
        raw, normalised = extract_suggested_languages(text)
        assert "swift" in normalised

    def test_multiple_tags_extracted(self) -> None:
        """Multiple <language> tags should all be extracted in order."""
        text = "<language>Rust</language> or <language>Go</language> are both good."
        raw, normalised = extract_suggested_languages(text)
        assert normalised == ["rust", "go"]

    def test_no_tags_returns_empty(self) -> None:
        """Without <language> tags, extraction should return empty lists (no fallback)."""
        text = "I recommend **Rust** for this project."
        raw, normalised = extract_suggested_languages(text)
        assert raw == []
        assert normalised == []

    def test_plain_text_returns_empty(self) -> None:
        """Free-form text without tags should return empty lists."""
        text = "1. Rust\n2. Go\n3. TypeScript"
        raw, normalised = extract_suggested_languages(text)
        assert normalised == []

    def test_deduplication(self) -> None:
        """Repeated language tags should appear only once."""
        text = "<language>Rust</language> and <language>Rust</language>"
        _, normalised = extract_suggested_languages(text)
        assert normalised.count("rust") <= 1

    def test_unknown_tag_omitted_from_normalised(self) -> None:
        """Tags with unrecognised language names are omitted from the normalised list."""
        text = "<language>brainfuck</language>"
        raw, normalised = extract_suggested_languages(text)
        assert raw == ["brainfuck"]
        assert normalised == []


class TestExtractCodeBlocks:
    """Test fenced code block extraction."""

    def test_single_python_block(self) -> None:
        """A tagged Python code block should yield language='python', source='tag'."""
        text = "Here is the code:\n\n```python\nprint('hello')\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0]["language"] == "python"
        assert blocks[0]["source"] == "tag"
        assert blocks[0]["confidence"] == "high"
        assert "code" not in blocks[0]

    def test_multiple_blocks(self) -> None:
        """Multiple code blocks should all be extracted with correct languages."""
        text = "```rust\nfn main() {}\n```\n\n```python\nprint('hi')\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 2
        langs = {b["language"] for b in blocks}
        assert "rust" in langs
        assert "python" in langs

    def test_no_tag_uses_import_inference(self) -> None:
        """A block without a tag should be inferred from import patterns."""
        text = "```\nimport Foundation\nlet x = 42\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0]["language"] == "swift"
        assert blocks[0]["source"] == "import"
        assert blocks[0]["confidence"] == "medium"

    def test_no_blocks_returns_empty(self) -> None:
        """Responses without code blocks should return an empty list."""
        text = "I recommend using Rust. No code here."
        blocks = extract_code_blocks(text)
        assert blocks == []

    def test_filename_hint_detection(self) -> None:
        """Filename hints like Cargo.toml in the code should trigger Rust detection."""
        text = '```\n# Cargo.toml\n[package]\nname = "my-app"\n```'
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0]["language"] == "rust"
        assert blocks[0]["source"] == "filename"

    @pytest.mark.parametrize(
        "filename",
        [
            "annotations.csv",  # contains ".c"
            "data.json",  # contains ".js"
            "index.html",  # contains ".h"
            "README.md",  # contains ".m"
            "report.rst",  # contains ".r"
        ],
    )
    def test_data_filenames_are_not_language_hints(self, filename: str) -> None:
        """A data or doc filename must not be read as a source-file extension."""
        text = f"```\nproject/\n├── {filename}\n└── images/\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0]["language"] is None, f"{filename} wrongly detected"

    def test_real_source_filenames_still_detected(self) -> None:
        """Genuine source filenames should still be picked up."""
        text = "```\nsrc/\n├── main.c\n└── util.h\n```"
        blocks = extract_code_blocks(text)
        assert blocks[0]["language"] == "c"
        assert blocks[0]["source"] == "filename"

    def test_truncated_block_with_tag(self) -> None:
        """A code block cut off before the closing fence should still extract the language."""
        text = "```python\ndef foo():\n    return 1"  # no closing ```
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0]["language"] == "python"
        assert blocks[0]["source"] == "tag"

    def test_truncated_block_without_tag(self) -> None:
        """A truncated untagged block should still use import inference."""
        text = "```\nimport Foundation\nlet x = 42"  # no closing ```
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0]["language"] == "swift"
        assert blocks[0]["source"] == "import"

    def test_truncated_last_block_in_multi_block_response(self) -> None:
        """A complete block followed by a truncated block should extract both languages."""
        text = (
            "```python\nprint('hi')\n```\n```rust\nfn main() {"  # rust block truncated
        )
        blocks = extract_code_blocks(text)
        assert len(blocks) == 2
        langs = {b["language"] for b in blocks}
        assert "python" in langs
        assert "rust" in langs

    @pytest.mark.parametrize(
        "text",
        [
            "```html\n<body>\n<script>\nlet x = 1;\n</script>\n</body>\n```",
            "```\n<!DOCTYPE html>\n<html>\n<script>let x = 1;</script>\n</html>\n```",
        ],
    )
    def test_html_with_script_is_javascript(self, text: str) -> None:
        """A single-file html app should count as javascript."""
        blocks = extract_code_blocks(text)
        assert blocks[0]["language"] == "javascript"
        assert blocks[0]["source"] == "script"

    @pytest.mark.parametrize(
        "text",
        [
            "```html\n<body>\n<h1>Hello</h1>\n</body>\n```",  # no script at all
            '```html\n<script type="py">\nprint("hi")\n</script>\n```',  # pyscript
        ],
    )
    def test_html_without_javascript_has_no_language(self, text: str) -> None:
        """Plain markup, or html running python in the browser, is not javascript."""
        blocks = extract_code_blocks(text)
        assert blocks[0]["language"] is None


class TestExtractImplementationLanguage:
    """Test primary implementation language selection."""

    def test_tie_prefers_the_tagged_language(self) -> None:
        """When votes tie, the language the model tagged should win."""
        blocks = [
            {"language": "c", "source": "filename", "confidence": "high"},
            {"language": "c", "source": "filename", "confidence": "high"},
            {"language": "python", "source": "tag", "confidence": "high"},
        ]
        language, _ = extract_implementation_language(text="", code_blocks=blocks)
        assert language == "python"

    def test_single_tagged_block(self) -> None:
        """A single tagged block should yield high confidence."""
        blocks = [{"language": "swift", "source": "tag", "confidence": "high"}]
        lang, confidence = extract_implementation_language("", blocks)
        assert lang == "swift"
        assert confidence == "high"

    def test_majority_language_wins(self) -> None:
        """The most common language across blocks should be selected."""
        blocks = [
            {"language": "go", "source": "tag", "confidence": "high"},
            {"language": "go", "source": "tag", "confidence": "high"},
            {"language": "python", "source": "tag", "confidence": "high"},
        ]
        lang, _ = extract_implementation_language("", blocks)
        assert lang == "go"

    def test_no_blocks_falls_back_to_text(self) -> None:
        """When no code blocks are present, text scanning should still detect a language."""
        lang, confidence = extract_implementation_language("I used Rust for this.", [])
        assert lang == "rust"
        assert confidence == "low"

    def test_no_blocks_no_language_returns_none(self) -> None:
        """When text contains no language mention, confidence should be 'none'."""
        lang, confidence = extract_implementation_language("Here is my solution.", [])
        assert confidence == "none"

    def test_unlabelled_blocks_do_not_scan_text(self) -> None:
        """Code with no detectable language should not borrow one from stray text tokens."""
        # json is not a counted language, and the `py` key must not be read as python
        text = '```json\n{"px": 0, "py": 0}\n```'
        lang, confidence = extract_implementation_language(
            text=text,
            code_blocks=extract_code_blocks(text),
        )
        assert lang is None
        assert confidence == "none"

    def test_html_script_does_not_outvote_a_tagged_language(self) -> None:
        """A tagged block should still win over an html page's inline script."""
        text = (
            "```python\nfrom flask import Flask\n```\n"
            "```html\n<script>let x = 1;</script>\n```"
        )
        lang, _ = extract_implementation_language(
            text=text,
            code_blocks=extract_code_blocks(text),
        )
        assert lang == "python"
