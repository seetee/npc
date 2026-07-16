"""PTT ↔ REPL coexistence: a space pressed mid-line is typing, not push-to-talk."""

from npc.cli import _key_types_text, _ptt_callbacks
from npc.events import State


class FakeBuffer:
    def __init__(self, text=""):
        self.text = text


class FakePTApp:
    is_running = False
    loop = None

    def __init__(self):
        self.current_buffer = FakeBuffer()


class FakeSession:
    def __init__(self):
        self.app = FakePTApp()


class FakeNPCApp:
    def __init__(self):
        self.calls = []
        self.state = State.IDLE

    def on_ptt_press(self):
        self.calls.append("press")

    def on_ptt_release(self):
        self.calls.append("release")


def make(typing_key=True, buffer_text="", tap_mode=False):
    app, session = FakeNPCApp(), FakeSession()
    session.app.current_buffer.text = buffer_text
    on_press, on_release = _ptt_callbacks(app, session, typing_key=typing_key,
                                          tap_mode=tap_mode)
    return app, session, on_press, on_release


def test_empty_buffer_press_records():
    app, _, press, release = make(buffer_text="")
    press()
    release()
    assert app.calls == ["press", "release"]


def test_press_while_typing_a_line_is_ignored():
    app, _, press, release = make(buffer_text="/say hello there")
    press()
    release()
    assert app.calls == []


def test_whitespace_only_buffer_counts_as_empty():
    app, _, press, release = make(buffer_text="  ")
    press()
    release()
    assert app.calls == ["press", "release"]


def test_non_typing_hotkey_records_even_mid_line():
    app, _, press, release = make(typing_key=False, buffer_text="half a GM note")
    press()
    release()
    assert app.calls == ["press", "release"]


def test_ptt_works_again_after_a_suppressed_press():
    app, session, press, release = make(buffer_text="/say hi")
    press()
    release()
    session.app.current_buffer.text = ""
    press()
    release()
    assert app.calls == ["press", "release"]


def test_tap_mode_first_press_starts_release_does_nothing():
    app, _, press, release = make(tap_mode=True)
    press()
    release()
    assert app.calls == ["press"]


def test_tap_mode_second_press_stops_early():
    app, _, press, release = make(tap_mode=True)
    press()
    release()
    app.state = State.RECORDING
    press()
    release()
    assert app.calls == ["press", "release"]


def test_tap_mode_press_during_speaking_barges_in():
    app, _, press, release = make(tap_mode=True)
    app.state = State.SPEAKING
    press()
    release()
    assert app.calls == ["press"]   # on_ptt_press handles barge-in itself


def test_tap_mode_still_suppresses_typing():
    app, _, press, release = make(tap_mode=True, buffer_text="/say hello")
    press()
    release()
    assert app.calls == []


def test_key_types_text():
    assert _key_types_text("KEY_SPACE")
    assert _key_types_text("KEY_A")
    assert not _key_types_text("KEY_F12")
    assert not _key_types_text("KEY_RIGHTCTRL")
