"""Roster discovery, name resolution, and per-NPC turn rendering."""

from pathlib import Path

from npc.config import Config
from npc.roster import (
    CharacterSlot,
    discover_character_files,
    load_slot,
    read_display_name,
    render_turns,
    resolve_npc,
)
from npc.session.history import ConversationHistory
from npc.session.logbook import Logbook


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_legacy_only_campaign(tmp_path):
    write(tmp_path / "character.md", "# Vess\n")
    files = discover_character_files(tmp_path)
    assert [(f.stem, f.legacy) for f in files] == [("character", True)]
    assert files[0].logbook_path == tmp_path / "logbook.md"


def test_characters_dir_only_sorted(tmp_path):
    write(tmp_path / "characters" / "mira.md", "# Mira\n")
    write(tmp_path / "characters" / "korval.md", "# Korval the Red\n")
    files = discover_character_files(tmp_path)
    assert [f.stem for f in files] == ["korval", "mira"]
    assert files[0].logbook_path == tmp_path / "logbooks" / "korval.md"
    assert not files[0].legacy


def test_legacy_comes_first_in_union(tmp_path):
    write(tmp_path / "character.md", "# Vess\n")
    write(tmp_path / "characters" / "korval.md", "# Korval\n")
    assert [f.stem for f in discover_character_files(tmp_path)] == \
        ["character", "korval"]


def test_stem_collision_skips_characters_character_md(tmp_path):
    write(tmp_path / "character.md", "# Vess\n")
    write(tmp_path / "characters" / "character.md", "# Impostor\n")
    files = discover_character_files(tmp_path)
    assert len(files) == 1
    assert files[0].legacy


def test_non_markdown_files_ignored(tmp_path):
    write(tmp_path / "characters" / "notes.txt", "not a character")
    write(tmp_path / "characters" / "korval.md", "# Korval\n")
    assert [f.stem for f in discover_character_files(tmp_path)] == ["korval"]


def test_empty_campaign_gives_empty_roster(tmp_path):
    assert discover_character_files(tmp_path) == []


def test_display_name_falls_back_to_stem():
    assert read_display_name("# Korval the Red\n\nText.", "korval") == "Korval the Red"
    assert read_display_name("No heading here.", "korval") == "korval"


def test_load_slot_reads_voice_mapping(tmp_path):
    write(tmp_path / "characters" / "korval.md", "# Korval the Red\n")
    config = Config(campaign_dir=tmp_path)
    config.tts.voices = {"korval": "en_GB-test"}
    ref = discover_character_files(tmp_path)[0]
    slot = load_slot(ref, config)
    assert slot.name == "Korval the Red"
    assert slot.voice == "en_GB-test"
    assert slot.history.limit == config.history_limit
    assert slot.logbook.path == tmp_path / "logbooks" / "korval.md"


def test_refresh_keeps_conversation_state(tmp_path):
    write(tmp_path / "characters" / "korval.md", "# Korval\n")
    config = Config(campaign_dir=tmp_path)
    slot = load_slot(discover_character_files(tmp_path)[0], config)
    slot.history.add_player("hello")
    slot.ooc_notes.append("be gruff")
    write(tmp_path / "characters" / "korval.md", "# Korval the Renamed\n")
    slot.refresh(config)
    assert slot.name == "Korval the Renamed"
    assert len(slot.history) == 1
    assert slot.ooc_notes == ["be gruff"]


def make_roster(*names):
    roster = {}
    for stem, name in names:
        roster[stem] = CharacterSlot(
            stem=stem, path=Path(f"{stem}.md"), name=name, character="",
            logbook=Logbook(Path(f"{stem}-log.md")),
            history=ConversationHistory())
    return roster


def test_resolve_exact_beats_prefix():
    roster = make_roster(("mira", "Mira"), ("mirabel", "Mirabel"))
    assert resolve_npc(roster, "mira").stem == "mira"
    assert resolve_npc(roster, "MIRABEL").stem == "mirabel"


def test_resolve_unique_prefix_and_display_name():
    roster = make_roster(("character", "Vess of the Amber Monolith"),
                         ("korval", "Korval the Red"))
    assert resolve_npc(roster, "kor").stem == "korval"
    assert resolve_npc(roster, "vess of").stem == "character"


def test_resolve_ambiguous_and_unknown():
    roster = make_roster(("mira", "Mira"), ("mirabel", "Mirabel"))
    assert len(resolve_npc(roster, "mir")) == 2   # ambiguous → candidates
    assert resolve_npc(roster, "zorg") == []      # unknown → empty list


def test_render_turns_matches_transcript_shape():
    turns = [("PLAYER", "who are you?"), ("Vess", "A keeper of secrets."),
             ("GM", "be more hostile")]
    assert render_turns(turns) == (
        "**PLAYER:** who are you?\n\n**Vess:** A keeper of secrets.\n\n"
        "**GM:** be more hostile")
