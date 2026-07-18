"""Onboarding UX: the init wizard, quick-start lines, forgiving commands."""

import textwrap

from npc.app import NPCApp
from npc.cli import (
    init_campaign,
    quickstart,
    run_init_wizard,
    skeleton_character,
    skeleton_secrets,
)
from npc.config import load_config
from npc.events import Info
from npc.session.secrets import SecretsSheet


class Script:
    """Scripted `ask` for the wizard; records the prompts it was shown."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else ""


def test_wizard_enter_keeps_the_example_npc(tmp_path):
    out = []
    created = run_init_wizard(tmp_path / "c", ask=Script(""), out=out.append)
    names = [p.name for p in created]
    assert "character.md" in names and "secrets.md" in names
    text = (tmp_path / "c" / "character.md").read_text(encoding="utf-8")
    assert "Vess of the Glass Monolith" in text
    printed = "\n".join(out)
    assert "npc doctor --fix" in printed and "npc run" in printed


def test_wizard_custom_npc_personalizes_files(tmp_path):
    script = Script("Elandra Kask", "a sly harbor-mistress")
    run_init_wizard(tmp_path / "c", ask=script, out=lambda s: None)
    sheet_text = (tmp_path / "c" / "character.md").read_text(encoding="utf-8")
    assert sheet_text.startswith("# Elandra Kask\n")
    assert "a sly harbor-mistress." in sheet_text
    secrets_text = (tmp_path / "c" / "secrets.md").read_text(encoding="utf-8")
    assert "Elandra Kask" in secrets_text
    # adventure/config/logbook still come from the templates
    assert (tmp_path / "c" / "adventure.md").exists()
    assert (tmp_path / "c" / "config.toml").exists()


def test_wizard_never_overwrites(tmp_path):
    (tmp_path / "character.md").write_text("# Mine\n", encoding="utf-8")
    run_init_wizard(tmp_path, ask=Script("Other Name", "someone"),
                    out=lambda s: None)
    assert (tmp_path / "character.md").read_text(encoding="utf-8") == "# Mine\n"


def test_skeleton_secrets_is_inert_until_armed():
    """The wizard's secrets.md must parse as ZERO secrets, and its indented
    example must be a VALID secret once dedented — the doc teaches by example,
    so the example has to actually work."""
    text = skeleton_secrets("Elandra")
    assert SecretsSheet.parse(text).entries == []
    example = textwrap.dedent(
        "\n".join(line[4:] if line.startswith("    ") else line
                  for line in text.splitlines()
                  if line.startswith("    ") or not line.strip()))
    armed = SecretsSheet.parse(example)
    assert [s.id for s in armed.entries] == ["harbor-ledger"]
    assert armed.entries[0].hint.startswith("who really pays")


def test_skeleton_character_carries_name_and_concept():
    text = skeleton_character("Bro Ulv", "en tystlåten smed från norr")
    assert text.startswith("# Bro Ulv\n")
    assert "en tystlåten smed från norr." in text
    assert "## Hard rules" in text


def app_for(campaign):
    from test_app_pipeline import FakeLLM

    return NPCApp(load_config(campaign), llm=FakeLLM(), on_event=lambda e: None)


def test_quickstart_mentions_only_what_applies(campaign):
    app = app_for(campaign)
    config = load_config(campaign)
    text = quickstart(app, config, voice_on=True)
    assert "hold" in text
    assert config.hotkey.key.removeprefix("KEY_").lower() in text
    assert "2 gated clues" in text          # template secrets.md ships two
    assert "/npc" not in text               # single-NPC campaign

    text = quickstart(app, config, voice_on=False)
    assert "mic off" in text and "hold" not in text


def test_quickstart_multi_npc_line(campaign):
    (campaign / "characters").mkdir()
    (campaign / "characters" / "korval.md").write_text("# Korval\n",
                                                       encoding="utf-8")
    app = app_for(campaign)
    text = quickstart(app, load_config(campaign), voice_on=True)
    assert "/npc <name>       switch NPC (2 in" in text


def test_unknown_command_suggests_closest(campaign):
    events = []
    app = app_for(campaign)
    app._on_event = events.append
    app.handle_line("/staus")
    message = [e for e in events if isinstance(e, Info)][-1].message
    assert "did you mean /status?" in message


def test_init_campaign_overrides(tmp_path):
    init_campaign(tmp_path, {"character.md": "# Custom\n"})
    assert (tmp_path / "character.md").read_text(encoding="utf-8") == "# Custom\n"
    assert "Vess" in (tmp_path / "adventure.md").read_text(encoding="utf-8")


def test_quickstart_lore_line_only_when_present(campaign):
    config = load_config(campaign)
    app = app_for(campaign)
    assert "lore" not in quickstart(app, config, voice_on=True)

    lore = campaign / "lore"
    lore.mkdir()
    (lore / "region.txt").write_text("word " * 500, encoding="utf-8")
    app = app_for(campaign)
    text = quickstart(app, config, voice_on=True)
    assert "~500 words of reference loaded" in text


def test_is_loopback_truth_table():
    from npc.cli import is_loopback

    for host in ("127.0.0.1", "127.0.0.53", "localhost", "LOCALHOST", "::1"):
        assert is_loopback(host), host
    for host in ("0.0.0.0", "192.168.1.23", "10.0.0.5", "::"):
        assert not is_loopback(host), host


def test_overlay_announcement_lines():
    from npc.cli import overlay_announcement

    assert overlay_announcement("127.0.0.1", 8765) == \
        ["overlay: http://127.0.0.1:8765"]
    lan = overlay_announcement("192.168.1.23", 8765)
    assert lan[0].startswith("overlay (LAN): http://192.168.1.23:8765")
    assert any("EVERYONE on this network" in line for line in lan)
    assert any("never leave this machine" in line for line in lan)
    # 0.0.0.0 resolves to a concrete address for the tablet, never 0.0.0.0
    wild = overlay_announcement("0.0.0.0", 8765)
    assert "0.0.0.0" not in wild[0]
