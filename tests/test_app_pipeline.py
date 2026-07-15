"""Full pipeline with fakes injected — no audio hardware, no Ollama."""

import numpy as np
import pytest

from npc.app import NPCApp, State
from npc.audio.recorder import AudioClip


class FakeLLM:
    model = "fake:latest"

    def __init__(self):
        self.calls = []

    def chat(self, system, messages):
        self.calls.append((system, messages))
        return "Greetings, traveler."

    def summarize_session(self, transcript, logbook_tail):
        return f"**Location:** the docks\n(summary of {len(transcript)} chars)"


class FakeTranscriber:
    def __init__(self, text="who are you?"):
        self.text = text

    def transcribe(self, clip):
        return self.text


class FakeRecorder:
    def __init__(self, seconds=1.0):
        self.on_auto_stop = None
        self.seconds = seconds

    def start(self):
        pass

    def stop(self):
        return AudioClip(np.zeros(int(16000 * self.seconds), dtype=np.int16))


class FakeSpeaker:
    def __init__(self):
        self.spoken = []
        self.stopped = 0

    def say(self, text):
        self.spoken.append(text)

    def stop(self):
        self.stopped += 1


@pytest.fixture
def app(config):
    lines = []
    app = NPCApp(config, llm=FakeLLM(), transcriber=FakeTranscriber(),
                 recorder=FakeRecorder(), speaker=FakeSpeaker(),
                 out=lines.append)
    app.output = lines
    app.start()
    yield app
    app._queue.put(None)


def drain(app):
    app._queue.join()


def test_voice_turn_end_to_end(app):
    app.on_ptt_press()
    app.on_ptt_release()
    drain(app)

    assert app.speaker.spoken == ["Greetings, traveler."]
    assert any("who are you?" in line for line in app.output)
    assert any("Greetings, traveler." in line for line in app.output)
    # NPC name parsed from template character sheet's first heading
    assert app.npc_name == "Vess of the Amber Monolith"
    # transcript written to disk
    content = app.transcript.read()
    assert "**PLAYER:** who are you?" in content
    assert "**NPC:** Greetings, traveler." in content
    assert app.state is State.IDLE


def test_voice_is_in_character_and_typed_is_ooc(app):
    app.handle_line("be more hostile")
    app.on_ptt_press()
    app.on_ptt_release()
    drain(app)

    system, messages = app.llm.calls[-1]
    assert "- be more hostile" in system            # standing instruction
    assert any(m["content"].startswith("GM NOTE") for m in messages)
    assert any(m["content"].startswith("PLAYER (spoken)") for m in messages)


def test_say_command_is_in_character(app):
    assert app.handle_line("/say I offer you this shin") == "ok"
    drain(app)
    _, messages = app.llm.calls[-1]
    assert messages[-1]["content"] == 'PLAYER (spoken): "I offer you this shin"'
    assert app.speaker.spoken == ["Greetings, traveler."]


def test_too_short_clip_discarded(app):
    app.recorder.seconds = 0.1
    app.on_ptt_press()
    app.on_ptt_release()
    drain(app)
    assert app.llm.calls == []
    assert any("too short" in line for line in app.output)
    assert app.state is State.IDLE


def test_busy_while_processing(app):
    app._state = State.PROCESSING
    app.on_ptt_press()
    assert any("busy" in line for line in app.output)
    app._state = State.IDLE


def test_barge_in_stops_playback(app):
    app._state = State.SPEAKING
    app.on_ptt_press()
    assert app.speaker.stopped == 1
    assert app.state is State.RECORDING
    app.on_ptt_release()
    drain(app)


def test_end_writes_logbook(app, config):
    app.handle_line("/say hello")
    drain(app)
    assert app.handle_line("/end") == "end"
    app.shutdown(summarize=True)

    text = config.logbook_file.read_text()
    assert "## Session 1" in text
    assert "**Location:** the docks" in text

    # a fresh app for the next session sees the summary and numbers itself 2
    app2 = NPCApp(config, llm=FakeLLM(), out=lambda s: None)
    assert app2.session_no == 2
    system = None
    app2.history.add_player("hi")
    from npc.session.prompt import build_system_prompt
    system = build_system_prompt(app2.character, app2.adventure,
                                 app2.logbook.tail(3), [])
    assert "**Location:** the docks" in system


def test_quit_does_not_summarize(app, config):
    app.handle_line("/say hello")
    drain(app)
    assert app.handle_line("/quit") == "quit"
    app.shutdown(summarize=False)
    assert "## Session" not in config.logbook_file.read_text()


def test_reload_picks_up_edits(app, config):
    config.character_file.write_text("# Renamed One\n\nA new soul.")
    app.handle_line("/reload")
    assert app.npc_name == "Renamed One"


def test_reload_applies_llm_model_from_config(app, config):
    (config.campaign_dir / "config.toml").write_text(
        'history_limit = 12\n[llm]\nmodel = "llama3.1:8b"\n[stt]\nmodel = "medium"\n'
    )
    app.handle_line("/reload")
    assert app.llm.model == "llama3.1:8b"          # applied live
    assert app.history.limit == 12
    assert any("after restart" in line for line in app.output)   # stt change flagged


def test_reload_survives_broken_config(app, config):
    (config.campaign_dir / "config.toml").write_text("[llm\nbroken")
    app.handle_line("/reload")
    assert any("not reloaded" in line for line in app.output)
    assert app.llm.model == "fake:latest"


def test_checkpoint_every_n_turns(config):
    config.checkpoint_every_turns = 2
    app = NPCApp(config, llm=FakeLLM(), out=lambda s: None)
    app.start()
    app.handle_line("/say one")
    app.handle_line("/say two")
    app._queue.join()
    app._queue.put(None)
    assert "## Session 1" in config.logbook_file.read_text()
