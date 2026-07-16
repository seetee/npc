import pytest

from npc.config import ConfigError, load_config


def test_defaults_without_config_file(tmp_path):
    config = load_config(tmp_path)
    assert config.llm.model == "qwen2.5:7b-instruct"
    assert config.stt.language == "auto"
    assert config.tts.voice == "en_GB-alba-medium"
    assert config.hotkey.key == "KEY_SPACE"
    assert config.history_limit == 30


def test_overrides(tmp_path):
    (tmp_path / "config.toml").write_text(
        "history_limit = 10\n"
        '[llm]\nmodel = "llama3.1:8b"\n'
        '[stt]\nlanguage = "sv"\n'
        '[hotkey]\nkey = "KEY_F12"\ngrab = true\n'
    )
    config = load_config(tmp_path)
    assert config.llm.model == "llama3.1:8b"
    assert config.stt.language == "sv"
    assert config.hotkey.key == "KEY_F12"
    assert config.hotkey.grab is True
    assert config.history_limit == 10
    # untouched sections keep defaults
    assert config.tts.voice == "en_GB-alba-medium"


def test_invalid_hotkey_mode_rejected(tmp_path):
    (tmp_path / "config.toml").write_text('[hotkey]\nmode = "double-tap"\n')
    with pytest.raises(ConfigError, match='"hold" or "tap"'):
        load_config(tmp_path)


def test_unknown_key_is_a_friendly_error(tmp_path):
    (tmp_path / "config.toml").write_text("[llm]\nmodell = 'oops'\n")
    with pytest.raises(ConfigError, match="valid keys"):
        load_config(tmp_path)


def test_env_var_overrides_config_api_key(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text('[llm]\napi_key = "from-file"\n')
    assert load_config(tmp_path).llm.api_key == "from-file"
    monkeypatch.setenv("NPC_LLM_API_KEY", "from-env")
    assert load_config(tmp_path).llm.api_key == "from-env"


def test_voice_path_expands_user(tmp_path):
    (tmp_path / "config.toml").write_text('[tts]\nvoices_dir = "~/voices"\n')
    config = load_config(tmp_path)
    assert "~" not in str(config.tts.voice_path)
    assert config.tts.voice_path.name == "en_GB-alba-medium.onnx"
