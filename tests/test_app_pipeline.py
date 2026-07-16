"""Full pipeline with fakes injected — no audio hardware, no LLM server."""

import numpy as np
import pytest

from npc.app import NPCApp, State
from npc.audio.recorder import AudioClip
from npc.events import (
    Busy,
    ConfigReloaded,
    ErrorOccurred,
    LogbookWritten,
    NpcReplied,
    NpcReplyChunk,
    PlayerSpoke,
    RecordingDiscarded,
    StateChanged,
    StatusReport,
    TurnCompleted,
)
from npc.llm import StreamingNotSupported


class FakeLLM:
    model = "fake:latest"

    def __init__(self):
        self.calls = []
        self.reply = "Greetings, traveler."
        self.replies = []  # optional queue; falls back to .reply when empty

    def chat(self, system, messages):
        self.calls.append((system, messages))
        return self.replies.pop(0) if self.replies else self.reply

    def summarize_session(self, transcript, logbook_tail):
        return f"**Location:** the docks\n(summary of {len(transcript)} chars)"


class FakeTranscriber:
    def __init__(self, text="who are you?"):
        self.text = text
        self.calls = 0

    def transcribe(self, clip):
        self.calls += 1
        return self.text


class FakeRecorder:
    def __init__(self, seconds=1.0):
        self.on_auto_stop = None
        self.seconds = seconds
        self.silent = False  # True → all-zero samples (below any energy gate)

    def start(self):
        pass

    def stop(self):
        n = int(16000 * self.seconds)
        value = 0 if self.silent else 8000
        return AudioClip(np.full(n, value, dtype=np.int16))


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
    events = []
    app = NPCApp(config, llm=FakeLLM(), transcriber=FakeTranscriber(),
                 recorder=FakeRecorder(), speaker=FakeSpeaker(),
                 on_event=events.append)
    app.events = events
    app.start()
    yield app
    app._queue.put(None)


def drain(app):
    app._queue.join()


def of_type(app, event_type):
    return [e for e in app.events if isinstance(e, event_type)]


def test_voice_turn_end_to_end(app):
    app.on_ptt_press()
    app.on_ptt_release()
    drain(app)

    assert app.speaker.spoken == ["Greetings, traveler."]
    assert of_type(app, PlayerSpoke) == [PlayerSpoke("who are you?")]
    assert of_type(app, NpcReplied) == [
        NpcReplied("Vess of the Amber Monolith", "Greetings, traveler.")
    ]
    # transcript written to disk
    content = app.transcript.read()
    assert "**PLAYER:** who are you?" in content
    assert "**NPC:** Greetings, traveler." in content
    assert app.state is State.IDLE


def test_state_events_trace_the_turn(app):
    """The StateChanged stream is what an OBS overlay would subscribe to."""
    app.on_ptt_press()
    app.on_ptt_release()
    drain(app)
    states = [e.state for e in of_type(app, StateChanged)]
    assert states == [State.RECORDING, State.PROCESSING, State.SPEAKING, State.IDLE]


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


def test_llm_failure_becomes_error_event_and_session_stays_usable(app):
    from npc.llm import LlmError

    real_chat = app.llm.chat

    def boom(system, messages):
        raise LlmError("cannot reach the LLM server at http://localhost:11434")

    app.llm.chat = boom
    app.handle_line("/say hello?")
    drain(app)
    assert any("cannot reach" in e.message for e in of_type(app, ErrorOccurred))
    assert app.state is State.IDLE

    app.llm.chat = real_chat                       # server "comes back"
    app.handle_line("/say hello again")
    drain(app)
    assert app.speaker.spoken == ["Greetings, traveler."]


def test_llm_decoration_is_stripped_before_speaking(app):
    app.llm.reply = ('Vess of the Amber Monolith: *adjusts her hood* '
                     '"Greetings, traveler." (smiles)')
    app.handle_line("/say hello")
    drain(app)
    assert app.speaker.spoken == ["Greetings, traveler."]
    assert of_type(app, NpcReplied)[-1].text == "Greetings, traveler."
    assert "**NPC:** Greetings, traveler." in app.transcript.read()


def test_too_short_clip_discarded(app):
    app.recorder.seconds = 0.1
    app.on_ptt_press()
    app.on_ptt_release()
    drain(app)
    assert app.llm.calls == []
    assert of_type(app, RecordingDiscarded) == [RecordingDiscarded("too short")]
    assert app.state is State.IDLE


def test_silent_clip_never_reaches_whisper(app):
    app.recorder.silent = True
    app.on_ptt_press()
    app.on_ptt_release()
    drain(app)
    assert app.transcriber.calls == 0              # energy gate short-circuits
    assert app.llm.calls == []
    assert of_type(app, RecordingDiscarded) == [RecordingDiscarded("silence")]
    assert app.state is State.IDLE


def test_phantom_transcript_discarded_before_llm(app):
    app.transcriber.text = "Thanks for watching!"
    app.on_ptt_press()
    app.on_ptt_release()
    drain(app)
    assert app.transcriber.calls == 1
    assert app.llm.calls == []
    discarded = of_type(app, RecordingDiscarded)
    assert len(discarded) == 1 and "hallucination" in discarded[0].reason
    assert app.state is State.IDLE


def test_auto_stop_enqueues_like_a_release(app):
    app.on_ptt_press()
    app.on_auto_stop(AudioClip(np.full(16000, 8000, dtype=np.int16)))
    drain(app)
    assert of_type(app, PlayerSpoke) == [PlayerSpoke("who are you?")]
    assert app.state is State.IDLE


def test_auto_stop_after_manual_release_is_a_noop(app):
    app.on_ptt_press()
    app.on_ptt_release()
    app.on_auto_stop(AudioClip(np.full(16000, 8000, dtype=np.int16)))
    drain(app)
    assert len(of_type(app, PlayerSpoke)) == 1     # exactly one turn


def test_too_short_auto_clip_discarded(app):
    app.on_ptt_press()
    app.on_auto_stop(AudioClip(np.full(800, 8000, dtype=np.int16)))  # 0.05 s
    drain(app)
    assert of_type(app, RecordingDiscarded) == [RecordingDiscarded("too short")]
    assert app.state is State.IDLE


def test_recording_started_flags_auto_stop_recorders(app):
    from npc.events import RecordingStarted

    app.recorder.is_auto_stop = True
    app.on_ptt_press()
    app.on_ptt_release()
    drain(app)
    assert of_type(app, RecordingStarted) == [RecordingStarted(auto_stop=True)]


def test_busy_while_processing(app):
    app._state = State.PROCESSING
    app.on_ptt_press()
    assert of_type(app, Busy)
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

    assert any(e.kind == "end" for e in of_type(app, LogbookWritten))
    text = config.logbook_file.read_text()
    assert "## Session 1" in text
    assert "**Location:** the docks" in text

    # a fresh app for the next session sees the summary and numbers itself 2
    app2 = NPCApp(config, llm=FakeLLM(), on_event=lambda e: None)
    assert app2.session_no == 2
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
    reloaded = of_type(app, ConfigReloaded)[-1]
    assert "[stt]" in reloaded.restart_needed      # stt change flagged


def test_reload_survives_broken_config(app, config):
    (config.campaign_dir / "config.toml").write_text("[llm\nbroken")
    app.handle_line("/reload")
    assert any("not reloaded" in e.message for e in of_type(app, ErrorOccurred))
    assert app.llm.model == "fake:latest"


def test_broken_event_subscriber_does_not_kill_the_session(config):
    def explode(event):
        raise RuntimeError("overlay crashed")

    app = NPCApp(config, llm=FakeLLM(), on_event=explode)
    app.start()
    app.handle_line("/say hello")
    app._queue.join()
    app._queue.put(None)
    assert len(app.llm.calls) == 1                 # turn still completed


# ---------- English-only lock ----------

def test_non_english_reply_is_reasked_in_english(app):
    app.llm.replies = ["Ja, jag talar flera tungomål, resenär.",
                       "Yes — I speak many tongues, traveler."]
    app.handle_line("/say kan du tala svenska?")
    drain(app)

    assert len(app.llm.calls) == 2
    correction = app.llm.calls[-1][1][-1]["content"]
    assert "English-only" in correction
    assert app.speaker.spoken == ["Yes — I speak many tongues, traveler."]
    assert of_type(app, NpcReplied)[-1].text == "Yes — I speak many tongues, traveler."


def test_reply_that_stays_foreign_after_retry_is_never_voiced(app):
    app.llm.replies = ["Ja, jag talar japanska.",
                       "はい、私には日本語ができます。なぜそう尋ねるのですか？"]
    app.handle_line("/say kan du tala japanska?")
    drain(app)

    assert len(app.llm.calls) == 2
    assert app.speaker.spoken == []                # Alba never gets CJK text
    assert "日本語" in of_type(app, NpcReplied)[-1].text  # but the GM sees it
    assert app.state is State.IDLE


def test_streaming_foreign_reply_never_reaches_tts(config):
    llm = StreamingFakeLLM(chunks=("Ja, jag talar gärna ",
                                   "ert tungomål, resenär."))
    llm.replies = ["Ja, jag talar ert tungomål, resenär.",
                   "Yes, I will speak your tongue."]
    events = []
    app = NPCApp(config, llm=llm, speaker=StreamingFakeSpeaker(),
                 on_event=events.append)
    app.events = events
    app.start()
    app.handle_line("/say tala svenska!")
    drain(app)
    app._queue.put(None)

    assert llm.stream_calls == 1
    assert len(llm.calls) == 2                     # fallback chat + English retry
    assert app.speaker.spoken == ["Yes, I will speak your tongue."]


def test_streaming_mid_reply_language_flip_is_skipped_from_audio(config):
    llm = StreamingFakeLLM(chunks=("You test my patience, traveler. ",
                                   "Ja, jag talar svenska, ju. Leave this place now."))
    events = []
    app = NPCApp(config, llm=llm, speaker=StreamingFakeSpeaker(),
                 on_event=events.append)
    app.events = events
    app.start()
    app.handle_line("/say hello")
    drain(app)
    app._queue.put(None)

    assert app.speaker.spoken == ["You test my patience, traveler.",
                                  "Leave this place now."]


# ---------- latency instrumentation ----------

def test_voice_turn_emits_timings(app):
    app.on_ptt_press()
    app.on_ptt_release()
    drain(app)
    turns = of_type(app, TurnCompleted)
    assert len(turns) == 1
    t = turns[0]
    assert t.stt_seconds is not None and t.stt_seconds >= 0
    assert t.llm_first_token_seconds is None       # non-streaming fakes
    assert t.speak_seconds is not None
    assert t.total_seconds >= t.llm_seconds >= 0


def test_say_turn_has_no_stt_timing(app):
    app.handle_line("/say hello")
    drain(app)
    assert of_type(app, TurnCompleted)[0].stt_seconds is None


def test_status_reports_last_and_average_turn(app):
    app.handle_line("/say one")
    drain(app)
    app.handle_line("/status")
    status = of_type(app, StatusReport)[-1]
    assert status.last_turn_seconds is not None
    assert status.avg_turn_seconds is not None


# ---------- streaming replies ----------

STREAM_CHUNKS = ("*bows* Greetings, tra", "veler. What brings", " you to the docks?")


class StreamingFakeLLM(FakeLLM):
    def __init__(self, chunks=STREAM_CHUNKS):
        super().__init__()
        self.chunks = chunks
        self.stream_calls = 0

    def chat_stream(self, system, messages):
        self.stream_calls += 1
        yield from self.chunks


class StreamingFakeSpeaker(FakeSpeaker):
    def __init__(self, cancel_after=None):
        super().__init__()
        self.cancel_after = cancel_after

    def say_stream(self, sentences):
        spoken = []
        for sentence in sentences:
            spoken.append(sentence)
            self.spoken.append(sentence)
            if self.cancel_after is not None and len(spoken) >= self.cancel_after:
                return spoken, True
        return spoken, False


@pytest.fixture
def stream_app(config):
    events = []
    app = NPCApp(config, llm=StreamingFakeLLM(), speaker=StreamingFakeSpeaker(),
                 on_event=events.append)
    app.events = events
    app.start()
    yield app
    app._queue.put(None)


def test_streaming_speaks_sentences_and_records_full_reply(stream_app):
    stream_app.handle_line("/say hello")
    drain(stream_app)

    assert stream_app.speaker.spoken == [
        "Greetings, traveler.",                    # *bows* never spoken
        "What brings you to the docks?",
    ]
    assert [e.text for e in of_type(stream_app, NpcReplyChunk)] == list(STREAM_CHUNKS)
    assert of_type(stream_app, NpcReplied) == [NpcReplied(
        "Vess of the Amber Monolith",
        "Greetings, traveler. What brings you to the docks?",
    )]
    assert stream_app.llm.calls == []              # non-streaming chat() never used
    assert ("**NPC:** Greetings, traveler. What brings you to the docks?"
            in stream_app.transcript.read())
    assert stream_app.state is State.IDLE


def test_streaming_turn_records_first_token_timing(stream_app):
    stream_app.handle_line("/say hello")
    drain(stream_app)
    t = of_type(stream_app, TurnCompleted)[0]
    assert t.llm_first_token_seconds is not None
    assert t.llm_first_token_seconds <= t.llm_seconds
    assert t.speak_seconds is not None


def test_streaming_narration_mix_speaks_only_quoted_dialogue(config):
    chunks = ('"You test my patience," I say, ',
              'my voice cold as the void. "Leave the shard."')
    events = []
    app = NPCApp(config, llm=StreamingFakeLLM(chunks), speaker=StreamingFakeSpeaker(),
                 on_event=events.append)
    app.events = events
    app.start()
    app.handle_line("/say give it to me")
    drain(app)
    app._queue.put(None)

    assert app.speaker.spoken == ["You test my patience.", "Leave the shard."]
    assert of_type(app, NpcReplied)[-1].text == "You test my patience. Leave the shard."


def test_barge_in_mid_stream_records_only_what_was_heard(config):
    events = []
    app = NPCApp(config, llm=StreamingFakeLLM(),
                 speaker=StreamingFakeSpeaker(cancel_after=1), on_event=events.append)
    app.events = events
    app.start()
    app.handle_line("/say hello")
    drain(app)
    app._queue.put(None)

    assert of_type(app, NpcReplied)[-1].text == "Greetings, traveler."
    content = app.transcript.read()
    assert "**NPC:** Greetings, traveler." in content
    assert "docks" not in content


def test_server_rejecting_stream_falls_back_to_plain_chat(config):
    class NoStreamLLM(FakeLLM):
        def chat_stream(self, system, messages):
            raise StreamingNotSupported("400: streaming not allowed")
            yield  # pragma: no cover — makes this a generator

    events = []
    app = NPCApp(config, llm=NoStreamLLM(), speaker=StreamingFakeSpeaker(),
                 on_event=events.append)
    app.events = events
    app.start()
    app.handle_line("/say hello")
    drain(app)
    app._queue.put(None)

    assert len(app.llm.calls) == 1                 # fell back to chat()
    assert app.speaker.spoken == ["Greetings, traveler."]
    assert of_type(app, NpcReplied)[-1].text == "Greetings, traveler."


def test_stream_false_in_config_uses_plain_chat(config):
    config.llm.stream = False
    app = NPCApp(config, llm=StreamingFakeLLM(), speaker=StreamingFakeSpeaker(),
                 on_event=lambda e: None)
    app.start()
    app.handle_line("/say hello")
    drain(app)
    app._queue.put(None)

    assert app.llm.stream_calls == 0
    assert len(app.llm.calls) == 1


def test_checkpoint_every_n_turns(config):
    config.checkpoint_every_turns = 2
    events = []
    app = NPCApp(config, llm=FakeLLM(), on_event=events.append)
    app.start()
    app.handle_line("/say one")
    app.handle_line("/say two")
    app._queue.join()
    app._queue.put(None)
    assert any(isinstance(e, LogbookWritten) and e.kind == "checkpoint" for e in events)
    assert "## Session 1" in config.logbook_file.read_text()
