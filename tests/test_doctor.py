"""doctor --fix: the interactive fixer loop (no network, no hardware)."""

import pytest

from npc.config import Config, TtsConfig
from npc.doctor import CheckResult, apply_fixes, npc_voice_checks


def test_apply_fixes_runs_only_confirmed_fixers():
    ran = []
    checks = [
        CheckResult("passing", True, fixer=lambda: ran.append("passing")),
        CheckResult("no-fixer", False, fix="sudo something  # print-only"),
        CheckResult("confirmed", False, fixer=lambda: ran.append("confirmed"),
                    fix_label="download the thing"),
        CheckResult("declined", False, fixer=lambda: ran.append("declined")),
    ]
    answers = iter(["y", "n"])
    out = []
    assert apply_fixes(checks, ask=lambda prompt: next(answers), out=out.append)
    assert ran == ["confirmed"]
    assert any("fixed: confirmed" in line for line in out)


def test_apply_fixes_survives_a_raising_fixer():
    def boom():
        raise RuntimeError("download failed")

    out = []
    assert apply_fixes([CheckResult("bad", False, fixer=boom, fix_label="x")],
                       ask=lambda prompt: "y", out=out.append)
    assert any("fix failed for bad" in line for line in out)


def test_apply_fixes_asks_nothing_when_all_pass():
    checks = [CheckResult("fine", True, fixer=lambda: None)]
    assert not apply_fixes(checks, ask=lambda prompt: pytest.fail("should not ask"),
                           out=lambda line: None)


def multi_config(tmp_path, voices):
    (tmp_path / "characters").mkdir()
    (tmp_path / "characters" / "korval.md").write_text("# Korval\n")
    return Config(campaign_dir=tmp_path,
                  tts=TtsConfig(voices_dir=str(tmp_path / "voices"), voices=voices))


def test_unknown_voice_mapping_stem_is_flagged(tmp_path):
    config = multi_config(tmp_path, {"korvall": "en_GB-x"})  # typo'd stem
    checks = npc_voice_checks(config)
    mapping = [c for c in checks if c.name.startswith("NPC voice mapping")]
    assert len(mapping) == 1 and not mapping[0].ok
    assert "korvall" in mapping[0].detail and "korval" in mapping[0].detail


def test_missing_mapped_voice_gets_a_fixer(tmp_path):
    config = multi_config(tmp_path, {"korval": "en_GB-extra-voice"})
    checks = npc_voice_checks(config)
    voice = [c for c in checks if c.name == "NPC voice (en_GB-extra-voice)"]
    assert len(voice) == 1 and not voice[0].ok
    assert voice[0].fixer is not None
    assert "en_GB-extra-voice" in voice[0].fix_label


def test_mapped_default_voice_is_not_rechecked(tmp_path):
    config = multi_config(tmp_path, {"korval": "en_GB-alba-medium"})  # = default
    assert [c for c in npc_voice_checks(config) if c.name.startswith("NPC voice (")] == []


def test_stem_collision_is_flagged(tmp_path):
    config = multi_config(tmp_path, {})
    (tmp_path / "character.md").write_text("# Vess\n")
    (tmp_path / "characters" / "character.md").write_text("# Impostor\n")
    collision = [c for c in npc_voice_checks(config) if c.name == "Character files"]
    assert len(collision) == 1 and not collision[0].ok


def test_secrets_check_passes_and_counts(tmp_path):
    from npc.doctor import npc_secrets_checks

    (tmp_path / "character.md").write_text("# Vess\n")
    (tmp_path / "secrets.md").write_text(
        "## a\nhint: h\n\nbody\n\n## b\nhint: h\nrevealed: session 1\n\nbody\n")
    checks = npc_secrets_checks(Config(campaign_dir=tmp_path))
    assert [(c.name, c.ok, c.detail) for c in checks] == [
        ("Secrets (character)", True, "1 locked, 1 revealed")]


def test_broken_secrets_file_is_flagged_soft(tmp_path):
    from npc.doctor import npc_secrets_checks

    (tmp_path / "character.md").write_text("# Vess\n")
    (tmp_path / "secrets.md").write_text("## bad\nno hint here\n")
    checks = npc_secrets_checks(Config(campaign_dir=tmp_path))
    assert len(checks) == 1 and not checks[0].ok and not checks[0].hard
    assert "missing its 'hint:'" in checks[0].detail


def test_missing_secrets_file_is_not_checked(tmp_path):
    from npc.doctor import npc_secrets_checks

    (tmp_path / "character.md").write_text("# Vess\n")
    assert npc_secrets_checks(Config(campaign_dir=tmp_path)) == []


def test_lore_check_counts_and_budget(tmp_path):
    from npc.doctor import npc_lore_checks

    (tmp_path / "character.md").write_text("# Vess\n", encoding="utf-8")
    lore = tmp_path / "lore"
    lore.mkdir()
    (lore / "small.txt").write_text("The river forks twice.", encoding="utf-8")
    config = Config(campaign_dir=tmp_path)
    checks = npc_lore_checks(config)
    assert [c.name for c in checks] == ["Lore (character)"]
    assert checks[0].ok and "1 file(s)" in checks[0].detail

    (lore / "big.txt").write_text("fact " * 20000, encoding="utf-8")
    checks = npc_lore_checks(config)
    budget = [c for c in checks if c.name == "Context budget (character)"]
    assert len(budget) == 1 and not budget[0].ok and not budget[0].hard
    assert "num_ctx = 32768" in budget[0].fix

    config.llm.num_ctx = 32768                      # following the advice
    assert not [c for c in npc_lore_checks(config)
                if c.name.startswith("Context budget")]


def test_lore_check_flags_thin_pdf(tmp_path):
    from test_lore import make_pdf

    from npc.doctor import npc_lore_checks

    (tmp_path / "character.md").write_text("# Vess\n", encoding="utf-8")
    lore = tmp_path / "lore"
    lore.mkdir()
    (lore / "thin.pdf").write_bytes(make_pdf("tiny"))
    checks = npc_lore_checks(Config(campaign_dir=tmp_path))
    thin = [c for c in checks if c.name == "Lore (character)"]
    assert len(thin) == 1 and not thin[0].ok
    assert "scanned/image PDF" in thin[0].detail
