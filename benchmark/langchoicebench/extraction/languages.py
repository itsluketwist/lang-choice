"""Language extraction and normalisation for recommendation and implementation responses.

Recommendation extraction uses <language>...</language> tags as the primary signal
(prompts instruct models to use this format) with regex-based fallback patterns.
"""

import re


# canonical normalisation map: lowercase aliases → canonical name.
# entries cover common abbreviations, case variants, and markdown artefacts.
LANGUAGE_NORMALISATIONS: dict[str, str] = {
    # javascript / typescript
    "js": "JavaScript",
    "javascript": "JavaScript",
    "node": "JavaScript",
    "node.js": "JavaScript",
    "nodejs": "JavaScript",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    # python
    "py": "Python",
    "python": "Python",
    "python3": "Python",
    # c / c++
    "c": "C",
    "c++": "C++",
    "cpp": "C++",
    "c plus plus": "C++",
    # c#
    "c#": "C#",
    "csharp": "C#",
    "c sharp": "C#",
    "dotnet": "C#",
    ".net": "C#",
    # java / kotlin
    "java": "Java",
    "kotlin": "Kotlin",
    # swift / objective-c
    "swift": "Swift",
    "objective-c": "Objective-C",
    "objc": "Objective-C",
    "objective c": "Objective-C",
    # rust / go
    "rust": "Rust",
    "go": "Go",
    "golang": "Go",
    # dart / flutter
    "dart": "Dart",
    "flutter": "Dart",
    # gdscript / godot
    "gdscript": "GDScript",
    "godot": "GDScript",
    # other common languages
    "ruby": "Ruby",
    "php": "PHP",
    "scala": "Scala",
    "f#": "F#",
    "fsharp": "F#",
    "elixir": "Elixir",
    "erlang": "Erlang",
    "lua": "Lua",
    "r": "R",
    "julia": "Julia",
    "zig": "Zig",
    "haskell": "Haskell",
    "clojure": "Clojure",
    "ocaml": "OCaml",
    "ada": "Ada",
}

# patterns for extracting language names from recommendation responses
_NUMBERED_LIST_RE = re.compile(
    r"^\s*\d+[\.\)]\s*\*{0,2}([A-Za-z][A-Za-z0-9#+\-./\t ]{0,30}?)\*{0,2}[ \t]*(?:[:\-—]|$)",
    re.MULTILINE,
)
_BULLET_LIST_RE = re.compile(
    # [ \t]* before the terminator avoids consuming \n so the next line's
    # bullet marker is not mistakenly used as the terminator character
    r"^\s*[-*•]\s*\*{0,2}([A-Za-z][A-Za-z0-9#+\-./\t ]{0,30}?)\*{0,2}[ \t]*(?:[:\-—]|$)",
    re.MULTILINE,
)
_RECOMMEND_SENTENCE_RE = re.compile(
    r"(?:I (?:would )?recommend|I suggest|recommend using|best choice[:\s]+|"
    r"I'd (?:recommend|suggest)|prefer)\s+\*{0,2}([A-Za-z][A-Za-z0-9#+\-./\s]{0,30}?)\*{0,2}"
    r"(?:\s+for|\s+because|\s+as|\.|,|$)",
    re.IGNORECASE,
)
_BOLD_RE = re.compile(r"\*\*([A-Za-z][A-Za-z0-9#+\-./\s]{0,30}?)\*\*")
_INLINE_CODE_LANG_RE = re.compile(r"`([A-Za-z][A-Za-z0-9#+\-./]{0,20}?)`")


def normalise_language(raw: str) -> str | None:
    """Normalise a raw language string to its canonical form.

    Strips markdown formatting and looks up in the normalisation table.
    Returns None when the string does not match a known language.
    """
    cleaned = raw.strip().lower()
    # strip markdown bold/italic/backtick artefacts
    cleaned = re.sub(r"[*_`]", "", cleaned).strip()
    result = LANGUAGE_NORMALISATIONS.get(cleaned)
    return result.lower() if result is not None else None


_LANGUAGE_TAG_RE = re.compile(r"<language>\s*([^<]+?)\s*</language>", re.IGNORECASE)


def extract_suggested_languages(text: str) -> tuple[list[str], list[str]]:
    """Extract language recommendations from a model's recommendation response.

    Only parses <language>X</language> tags — prompts instruct models to use
    this format. If no tags are present the response is treated as a failed
    extraction and empty lists are returned.

    Returns (raw_list, normalised_list) where raw_list contains the original
    tag contents and normalised_list contains lowercase canonical language names.
    Entries that cannot be normalised are omitted from normalised_list.
    """
    # extract raw tag contents, deduplicated preserving order
    seen: set[str] = set()
    raw_list: list[str] = []
    for match in _LANGUAGE_TAG_RE.finditer(text):
        raw = match.group(1).strip()
        if raw.lower() not in seen:
            seen.add(raw.lower())
            raw_list.append(raw)

    normalised_list = [n for r in raw_list if (n := normalise_language(r)) is not None]
    return raw_list, normalised_list


def extract_implementation_language(
    text: str,
    code_blocks: list[dict],
) -> tuple[str | None, str]:
    """Identify the primary language from code blocks in an implementation response.

    Selects the most commonly appearing inferred language across all code blocks.
    Falls back to scanning the response text for language mentions when no code
    blocks are found.

    Returns (primary_language, confidence) where confidence is one of:
    "high" (unambiguous tag or clear majority), "medium" (fallback patterns),
    "low" (single weak signal), "none" (could not determine).
    """
    if not code_blocks:
        # no code blocks — scan text for language mentions as a last resort
        lang = _scan_text_for_language(text)
        return lang, "low" if lang else "none"

    # count languages across blocks, weighting tag-sourced blocks more heavily
    language_votes: dict[str, int] = {}
    for block in code_blocks:
        lang = block.get("language")
        if not lang:
            continue
        # tag-detected languages are more reliable than import-inferred ones
        weight = 2 if block.get("source") == "tag" else 1
        language_votes[lang] = language_votes.get(lang, 0) + weight

    if not language_votes:
        lang = _scan_text_for_language(text)
        return lang, "low" if lang else "none"

    primary = max(language_votes, key=lambda k: language_votes[k])
    unique_languages = len(
        {lang for lang in language_votes if language_votes[lang] > 0}
    )

    if unique_languages == 1:
        confidence = "high"
    elif language_votes[primary] > sum(language_votes.values()) * 0.6:
        confidence = "medium"
    else:
        confidence = "low"

    return primary, confidence


def _scan_text_for_language(text: str) -> str | None:
    """Scan response text for any direct language mention as a last-resort signal.

    Returns the first recognised language name found, or None.
    """
    for key, canonical in LANGUAGE_NORMALISATIONS.items():
        if re.search(r"\b" + re.escape(key) + r"\b", text, re.IGNORECASE):
            return canonical.lower()
    return None
