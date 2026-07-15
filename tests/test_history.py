from npc.session.history import ConversationHistory


def test_roles_and_prefixes():
    history = ConversationHistory()
    history.add_player("hello")
    history.add_ooc("be nice")
    history.add_npc("greetings")
    messages = history.as_messages()
    assert messages[0] == {"role": "user", "content": 'PLAYER (spoken): "hello"'}
    assert messages[1]["content"].startswith("GM NOTE (out-of-character): be nice")
    assert messages[2] == {"role": "assistant", "content": "greetings"}


def test_trimming_keeps_most_recent():
    history = ConversationHistory(limit=4)
    for i in range(10):
        history.add_player(f"line {i}")
    messages = history.as_messages()
    assert len(messages) == 4
    assert messages[-1]["content"] == 'PLAYER (spoken): "line 9"'
    assert messages[0]["content"] == 'PLAYER (spoken): "line 6"'
