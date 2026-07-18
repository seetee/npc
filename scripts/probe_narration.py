"""Probe the NPC prompt against a real Ollama model for self-narration leaks.

Run after changing ROLE_FRAMING or swapping models:

    uv run python scripts/probe_narration.py [model]

Each turn prints the raw model output and, when the sanitizer changed it, the
cleaned version. RAW may still narrate occasionally (the sanitizer's job is to
catch it); the failure to watch for is narration surviving in CLEANED, or raw
QUOTELESS leading narration showing up often — that shape is only fixable in
the prompt, so iterate on ROLE_FRAMING if you see it in more than ~1 in 10
replies.
"""

import sys
from importlib import resources

from npc.llm import OllamaClient
from npc.session.prompt import build_system_prompt, extract_dialogue

NPC_NAME = "Vess of the Amber Monolith"
PROBES = [
    "What do you look like, old woman?",
    "Show me how you feel about the Amber Monolith.",
    "Tell me your saddest memory.",
    "We found a strange device in the ruins. Look at it.",
    "Are you angry with us? Show me.",
    "Varför bor du så nära monoliten?",
    "Beskriv dig själv för mig.",
]


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b-instruct"
    templates = resources.files("npc") / "templates"
    system = build_system_prompt(
        (templates / "character.md").read_text(encoding="utf-8"),
        (templates / "adventure.md").read_text(encoding="utf-8"),
        "", [],
    )
    llm = OllamaClient("http://localhost:11434", model)

    history: list[dict[str, str]] = []
    changed = 0
    for probe in PROBES:
        history.append({"role": "user", "content": f'PLAYER (spoken): "{probe}"'})
        raw = llm.chat(system, history)
        cleaned = extract_dialogue(raw, NPC_NAME)
        history.append({"role": "assistant", "content": cleaned})
        print(f"--- {probe}\nRAW    : {raw!r}")
        if cleaned != raw:
            changed += 1
            print(f"CLEANED: {cleaned!r}")
    print(f"\n{changed}/{len(PROBES)} replies needed cleaning "
          f"(inspect CLEANED lines above: they must be pure dialogue)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
