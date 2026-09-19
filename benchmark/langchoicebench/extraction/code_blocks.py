"""Extract fenced code blocks from markdown responses and infer their language.

Each block exposes {language, source, confidence} — raw code is not returned.
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


def _filename_pattern(key: str) -> re.Pattern[str]:
    """Build a regex that matches a filename hint as a whole token.

    A plain substring test is not enough: ".c" appears inside "annotations.csv",
    ".js" inside "data.json", and ".h" inside "index.html", which would otherwise
    report a confident language for a block that only lists data files.
    Returns the compiled pattern for one filename map key.
    """
    if key.startswith("."):
        # an extension must end the filename, so ".csv" no longer counts as ".c"
        return re.compile(rf"[\w-]+{re.escape(key)}(?![\w])")
    return re.compile(rf"\b{re.escape(key)}\b")


# longest keys first so ".cpp" is tried before ".c"
_FILENAME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_filename_pattern(key), lang)
    for key, lang in sorted(
        _FILENAME_LANGUAGE_MAP.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
]

# html documents: their program logic lives in inline <script> elements, unless
# the page runs python in the browser (pyscript, brython, pyodide)
_HTML_TAGS = {"html", "htm", "xhtml"}
_SCRIPT_RE = re.compile(r"<script\b", re.IGNORECASE)
_BROWSER_PYTHON_RE = re.compile(
    r"py-script|pyscript|brython|pyodide"
    r"|type=[\"'](?:text/)?(?:x-)?(?:python|py|mpy)[\"']",
    re.IGNORECASE,
)

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
    """Extract all triple-backtick fenced code blocks from a markdown response.

    Returns a list of {language, source, confidence} dicts, or empty if none found.
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

    Priority: explicit fence tag > html with inline script > filename hints >
    import/syntax patterns.
    """
    from langchoicebench.extraction.languages import normalise_language

    # 1. explicit fence tag — highest confidence
    if tag:
        normalised = normalise_language(tag)
        if normalised:
            return normalised, "tag", "high"

    # 2. a single-file html app is written in javascript. deliberately not a "tag"
    # source, so a block tagged with a real language still takes precedence.
    if _is_html_with_script(tag=tag, code=code):
        return "javascript", "script", "medium"

    # 3. filename hints — high confidence for unambiguous files
    lowered = code.lower()
    for pattern, lang in _FILENAME_PATTERNS:
        if pattern.search(lowered):
            return lang.lower(), "filename", "high"

    # 4. import/syntax patterns — medium confidence
    for pattern, lang in _IMPORT_LANGUAGE_PATTERNS:
        if pattern.search(code):
            return lang.lower(), "import", "medium"

    return None, None, None


def _is_html_with_script(
    tag: str,
    code: str,
) -> bool:
    """Check whether a block is an html document that runs javascript.

    The block must be tagged as html, or start like an html document, and
    contain a <script> element that is not running python in the browser.
    Returns True when the block's code should count as javascript.
    """
    is_html = tag.lower() in _HTML_TAGS or (
        not tag and code.lstrip().lower().startswith(("<!doctype html", "<html"))
    )
    return (
        is_html
        and _SCRIPT_RE.search(code) is not None
        and _BROWSER_PYTHON_RE.search(code) is None
    )
