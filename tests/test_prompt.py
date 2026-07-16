import pytest

from npc.session.prompt import build_system_prompt, extract_dialogue


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


def test_prompt_demands_voice_only_non_assistant_behavior():
    prompt = build_system_prompt("X", "", "", [])
    assert "spoken words" in prompt
    assert "not an assistant" in prompt


@pytest.mark.parametrize("raw, spoken", [
    ("Greetings, traveler.", "Greetings, traveler."),
    ("*strokes beard* Aye, that's the truth.", "Aye, that's the truth."),
    ("*He pauses.* Very well. *nods slowly*", "Very well."),
    ("(chuckles) You drive a hard bargain.", "You drive a hard bargain."),
    ("[sighs] Fine. Take it.", "Fine. Take it."),
    ("I hid it (glances at the door) under the floor.", "I hid it under the floor."),
    ('"Stay close to the lantern light."', "Stay close to the lantern light."),
    ("NPC: I know nothing of it.", "I know nothing of it."),
    ("Vess: The monolith hums at night.", "The monolith hums at night."),
    ("**Vess:** Ask the abbot.", "Ask the abbot."),
    # quotes inside the line are the character quoting something — keep them
    ('Well, "The Broken Crown" is what we call it.',
     'Well, "The Broken Crown" is what we call it.'),
])
def test_extract_dialogue_keeps_only_the_spoken_words(raw, spoken):
    assert extract_dialogue(raw, "Vess") == spoken


def test_pure_stage_direction_falls_back_to_plain_text():
    # nothing speakable survives — better an odd line than dead air at the table
    assert extract_dialogue("*stares at you in silence*", "Vess") == "stares at you in silence"
