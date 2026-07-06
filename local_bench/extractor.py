import re

# Fenced-block language aliases per task language. The model may tag a block
# with any of these and we still treat it as the target language.
_LANG_ALIASES = {
    "python": ["python", "py"],
    "typescript": ["typescript", "ts", "tsx", "javascript", "js"],
}

# Heuristic for "the whole reply is raw code" (no fences), per language.
_RAW_CODE_START = {
    "python": r"^(import |from |def |class |#|async def )",
    "typescript": r"^(import |export |const |let |function |class |//|type |interface |async )",
}


def _aliases(language: str) -> list[str]:
    return _LANG_ALIASES.get(language, [language])


def extract_code_blocks(text: str, language: str = "python") -> list[str]:
    """Extract fenced code blocks, with priority: language-tagged > any-tagged > raw code."""
    aliases = "|".join(re.escape(a) for a in _aliases(language))

    # Priority 1: blocks tagged with the target language (or an alias of it)
    lang_pattern = rf"```[ \t]*(?:{aliases})[ \t]*\n(.*?)```"
    lang_blocks = re.findall(lang_pattern, text, re.DOTALL | re.IGNORECASE)
    if lang_blocks:
        return [b.strip() for b in lang_blocks]

    # Priority 2: any fenced block (with or without a language tag)
    any_pattern = r"```(?:[^\n]*)?\n(.*?)```"
    any_blocks = re.findall(any_pattern, text, re.DOTALL)
    if any_blocks:
        return [b.strip() for b in any_blocks]

    # Priority 3: whole response looks like raw code
    stripped = text.strip()
    start = _RAW_CODE_START.get(language, r"^(import |from |def |class |#)")
    if stripped and re.match(start, stripped):
        return [stripped]

    return []


def extract_primary_code(text: str, language: str = "python") -> str:
    """Return all extracted code blocks joined together.

    Concatenates all blocks because models often split imports from implementation.
    """
    blocks = extract_code_blocks(text, language)
    if not blocks:
        return ""
    return "\n\n".join(blocks)
