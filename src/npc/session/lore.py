"""Per-NPC reference documents ("lore") injected into the system prompt.

Tier 1: whole documents, no retrieval. The budget is the model's context
window, so everything here reports word/token estimates that doctor and the
runtime warning check against `[llm] num_ctx`. Files load sorted by name so
the rendered block is byte-stable across turns — that keeps Ollama's KV
prefix cache warm (the lore block sits early in the prompt on purpose).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LORE_SUFFIXES = (".md", ".txt", ".pdf")


@dataclass(frozen=True)
class LoreFile:
    name: str  # filename, shown as the ## subsection heading
    text: str
    words: int
    pages: int = 0  # PDFs only; lets doctor spot image-only extractions


def estimate_tokens(text: str) -> int:
    """Rough Latin-text heuristic (~4 chars/token) — good enough for budget
    warnings; never used for hard limits."""
    return len(text) // 4


def _extract_pdf(path: Path) -> tuple[str, int]:
    from pypdf import PdfReader  # lazy: only campaigns with PDFs pay for it

    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text, len(reader.pages)


def load_lore(lore_dir: Path) -> tuple[list[LoreFile], list[str]]:
    """All lore files in a directory (non-recursive, sorted by name).
    Per-file failures land in the errors list — a broken PDF must never stop
    a session from starting."""
    files: list[LoreFile] = []
    errors: list[str] = []
    if not lore_dir.is_dir():
        return files, errors
    for path in sorted(lore_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in LORE_SUFFIXES:
            continue
        try:
            if path.suffix.lower() == ".pdf":
                text, pages = _extract_pdf(path)
            else:
                text, pages = path.read_text(encoding="utf-8"), 0
        except Exception as e:
            errors.append(f"{path.name}: {e}")
            continue
        text = text.strip()
        files.append(LoreFile(name=path.name, text=text,
                              words=len(text.split()), pages=pages))
    return files, errors


CTX_STEPS = (8192, 16384, 32768)


def suggest_num_ctx(tokens: int) -> int:
    """The num_ctx value doctor and the runtime warning recommend: the first
    standard step with ~25% headroom over the estimate."""
    for step in CTX_STEPS:
        if tokens * 1.25 <= step:
            return step
    return CTX_STEPS[-1]


def lore_block(files: list[LoreFile]) -> str:
    sections = "\n\n".join(f"## {f.name}\n{f.text}" for f in files if f.text)
    return (
        "# Reference knowledge\n"
        "\n"
        "Your character knows the following material deeply — it is\n"
        "established fact about the world. Rely on it when answering; never\n"
        "contradict it. When it does not cover something, say so in\n"
        "character instead of inventing details.\n"
        "\n" + sections
    )
