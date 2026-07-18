"""The examples/ gallery must always load clean — it's the shop window."""

from pathlib import Path

import pytest

from npc.config import load_config
from npc.roster import discover_character_files, load_slot

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").iterdir())


@pytest.mark.parametrize("campaign", EXAMPLES, ids=lambda p: p.name)
def test_example_campaign_loads_clean(campaign):
    config = load_config(campaign)
    refs = discover_character_files(campaign)
    assert refs, f"{campaign.name} has no character files"
    for ref in refs:
        slot = load_slot(ref, config)
        assert slot.secrets_error is None, slot.secrets_error
        assert slot.lore_errors == [], slot.lore_errors
        assert slot.name != slot.stem  # every sheet has a display-name heading


def test_gallery_shows_both_layouts():
    names = [p.name for p in EXAMPLES]
    assert "rusty-lantern" in names       # single legacy character.md
    assert "amber-monolith" in names      # characters/ multi-NPC layout
    assert any((p / "secrets.md").exists() or (p / "secrets").is_dir()
               for p in EXAMPLES)
    assert any((p / "lore").is_dir() for p in EXAMPLES)
