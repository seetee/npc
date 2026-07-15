from npc.session.prompt import build_system_prompt


def test_all_sections_in_order():
    prompt = build_system_prompt("CHARSHEET", "ADVENTURE", "LOGTAIL",
                                 ["be hostile", "players stole the idol"])
    expected_order = ["You are role-playing", "CHARSHEET", "ADVENTURE", "LOGTAIL",
                      "- be hostile", "- players stole the idol"]
    positions = [prompt.index(part) for part in expected_order]
    assert positions == sorted(positions)


def test_empty_sections_are_omitted():
    prompt = build_system_prompt("CHARSHEET", "", "", [])
    assert "CHARSHEET" in prompt
    assert "Adventure notes" not in prompt
    assert "Logbook" not in prompt
    assert "Standing GM instructions" not in prompt


def test_reply_language_is_english_even_for_swedish_input():
    prompt = build_system_prompt("X", "", "", [])
    assert "Always answer in English" in prompt
    assert "Swedish" in prompt
