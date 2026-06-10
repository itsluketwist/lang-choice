"""Extract code blocks from markdown responses and infer their programming language.

Handles triple-backtick fenced code blocks. Language detection uses the tag on
the fence, falls back to file-name hints and import patterns within the code.

Each extracted block exposes {language, source, confidence} — the raw code
content is not returned since the library only needs the language signal.
"""

import re


# regex for triple-backtick fenced code blocks with optional language tag
# group 1: language tag (may be empty), group 2: code content
_CODE_BLOCK_RE = re.compile(
    r"```(\w*)\n(.*?)(?:```|$)",
    re.DOTALL,
)

# file-name → language mappings used when inspecting code for common filenames
_FILENAME_LANGUAGE_MAP: dict[str, str] = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".rs": "Rust",
    ".go": "Go",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".c": "C",
    ".h": "C",
    ".cs": "C#",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".swift": "Swift",
    ".java": "Java",
    ".rb": "Ruby",
    ".php": "PHP",
    ".scala": "Scala",
    ".dart": "Dart",
    ".zig": "Zig",
    ".ex": "Elixir",
    ".exs": "Elixir",
    ".erl": "Erlang",
    ".fs": "F#",
    ".fsx": "F#",
    ".lua": "Lua",
    ".r": "R",
    ".jl": "Julia",
    ".m": "Objective-C",
    ".mm": "Objective-C",
    "cargo.toml": "Rust",
    "go.mod": "Go",
    "package.json": "JavaScript",
    "pom.xml": "Java",
    "build.gradle": "Kotlin",
    "cmakelists.txt": "C++",
}

# common import patterns that reveal language when the fence tag is missing.
# ordered from most specific to least specific — the first match wins.
# generic patterns (e.g. bare "import X") appear last to avoid false positives.
_IMPORT_LANGUAGE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # language-specific imports that cannot be confused with other languages
    (re.compile(r"^\s*use\s+std::", re.MULTILINE), "Rust"),
    (re.compile(r"^\s*fn\s+main\s*\(\s*\)", re.MULTILINE), "Rust"),
    (re.compile(r"^\s*#include\s+[<\"]", re.MULTILINE), "C++"),
    (re.compile(r"^\s*void\s+setup\s*\(\s*\)", re.MULTILINE), "C++"),
    (re.compile(r"^\s*package\s+main\s*$", re.MULTILINE), "Go"),
    (re.compile(r"^\s*import\s+\"", re.MULTILINE), "Go"),
    (re.compile(r"^\s*using\s+System", re.MULTILINE), "C#"),
    (re.compile(r"^\s*namespace\s+\w+", re.MULTILINE), "C#"),
    # swift-specific frameworks must precede generic `import X` patterns
    (re.compile(r"^\s*import\s+Foundation\b", re.MULTILINE), "Swift"),
    (re.compile(r"^\s*import\s+UIKit\b", re.MULTILINE), "Swift"),
    (re.compile(r"^\s*import\s+SwiftUI\b", re.MULTILINE), "Swift"),
    # kotlin / java (android.* imports are unambiguous)
    (re.compile(r"^\s*import\s+android\.", re.MULTILINE), "Kotlin"),
    (re.compile(r"^\s*fun\s+main\s*\(", re.MULTILINE), "Kotlin"),
    (re.compile(r"^\s*import\s+java\.", re.MULTILINE), "Java"),
    (re.compile(r"^\s*public\s+class\s+\w+", re.MULTILINE), "Java"),
    # javascript / typescript
    (re.compile(r"^\s*const\s+\w+\s*=\s*require\(", re.MULTILINE), "JavaScript"),
    (re.compile(r"^\s*import\s+\{", re.MULTILINE), "TypeScript"),
    (re.compile(r":\s*[A-Z][a-zA-Z]+\s*[=;{]", re.MULTILINE), "TypeScript"),
    # python — `from X import Y` is highly Python-specific; bare `import X` is too generic
    (re.compile(r"^\s*from\s+\w+\s+import\s+", re.MULTILINE), "Python"),
]


def extract_code_blocks(text: str) -> list[dict]:
    """Extract all fenced code blocks from a markdown response.

    Each returned dict contains:
      language   — canonical language name, or None if unidentifiable
      source     — how the language was determined: "tag", "filename", "import", or None
      confidence — detection confidence: "high", "medium", "low", or None

    The raw code content is intentionally not returned — the library only needs
    the language signal, not the code itself.

    Returns an empty list when no fenced blocks are present.
    """
    blocks = []
    for match in _CODE_BLOCK_RE.finditer(text):
        tag = match.group(1).strip()
        code = match.group(2)
        language, source, confidence = _infer_language(tag=tag, code=code)
        blocks.append(
            {
                "language": language,
                "source": source,
                "confidence": confidence,
            }
        )
    return blocks


def _infer_language(
    tag: str,
    code: str,
) -> tuple[str | None, str | None, str | None]:
    """Infer the programming language for a code block.

    Returns (language, source, confidence) where source indicates how the
    language was detected and confidence indicates reliability.

    Priority: explicit fence tag > filename hints > import/syntax patterns.
    """
    from langchoicebench.extraction.languages import normalise_language

    # 1. explicit fence tag — highest confidence
    if tag:
        normalised = normalise_language(tag)
        if normalised:
            return normalised, "tag", "high"

    # 2. filename hints — high confidence for unambiguous files
    for key, lang in _FILENAME_LANGUAGE_MAP.items():
        if key in code.lower():
            return lang.lower(), "filename", "high"

    # 3. import/syntax patterns — medium confidence
    for pattern, lang in _IMPORT_LANGUAGE_PATTERNS:
        if pattern.search(code):
            return lang.lower(), "import", "medium"

    return None, None, None
