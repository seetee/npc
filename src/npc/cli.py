"""Command-line entrypoint: init / run / doctor / transcribe / say."""

from __future__ import annotations

import argparse
import sys
from importlib import resources
from pathlib import Path

from .config import ConfigError, load_config

TEMPLATE_FILES = ("character.md", "adventure.md", "logbook.md", "config.toml")


def init_campaign(campaign_dir: Path) -> list[Path]:
    """Scaffold a campaign directory from templates; never overwrites."""
    campaign_dir.mkdir(parents=True, exist_ok=True)
    (campaign_dir / "sessions").mkdir(exist_ok=True)
    created = []
    templates = resources.files("npc") / "templates"
    for name in TEMPLATE_FILES:
        target = campaign_dir / name
        if not target.exists():
            target.write_text((templates / name).read_text(encoding="utf-8"),
                              encoding="utf-8")
            created.append(target)
    return created


def cmd_init(args) -> int:
    created = init_campaign(Path(args.campaign))
    if created:
        print(f"Campaign scaffolded in {Path(args.campaign).resolve()}:")
        for path in created:
            print(f"  {path.name}")
        print("\nEdit character.md (your NPC) and adventure.md, then: npc run "
              + args.campaign)
    else:
        print("All campaign files already exist — nothing created.")
    return 0


def cmd_doctor(args) -> int:
    from .doctor import apply_fixes, print_report, run_checks

    config = load_config(Path(args.campaign))
    print(f"Checking setup for {config.campaign_dir} …")
    checks = run_checks(config, deep=True)
    print_report(checks)
    if args.fix and not all(c.ok for c in checks):
        if not sys.stdin.isatty():
            print("--fix skipped: not an interactive terminal")
        elif apply_fixes(checks):
            print("\nRe-checking …")
            checks = run_checks(config, deep=True)
            print_report(checks)
    all_ok = all(c.ok for c in checks)
    print("\nAll good — ready to play." if all_ok
          else "\nFix the FAILs above (commands are copy-pasteable).")
    return 0 if all_ok else 1


def cmd_transcribe(args) -> int:
    import wave

    import numpy as np

    from .audio.recorder import AudioClip
    from .stt import WhisperTranscriber

    config = load_config(Path(args.campaign))
    with wave.open(args.file, "rb") as w:
        if w.getsampwidth() != 2:
            print("only 16-bit PCM wav supported", file=sys.stderr)
            return 1
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            frames = frames.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
        clip = AudioClip(samples=frames, sample_rate=w.getframerate())
    transcriber = WhisperTranscriber(config.stt.model, config.stt.language,
                                     config.stt.device)
    print(transcriber.transcribe(clip))
    return 0


def cmd_say(args) -> int:
    from .tts import PiperSpeaker, download_hint

    config = load_config(Path(args.campaign))
    voice_path = config.tts.voice_path
    if not voice_path.exists():
        print(f"voice not found: {voice_path}", file=sys.stderr)
        print(f"download it: {download_hint(config.tts.voice, voice_path.parent)}",
              file=sys.stderr)
        return 1
    PiperSpeaker(voice_path).say(args.text)
    return 0


def cmd_run(args) -> int:
    from .app import NPCApp
    from .doctor import print_report, run_checks
    from .events import NpcSwitched, TurnCompleted, format_timings, print_event
    from .llm import make_llm_client

    from .roster import discover_character_files

    config = load_config(Path(args.campaign))
    if not discover_character_files(config.campaign_dir):
        print(f"no character.md or characters/*.md in {config.campaign_dir} — "
              f"run: npc init {args.campaign}", file=sys.stderr)
        return 1

    checks = run_checks(config, deep=False)
    if not print_report(checks):
        print("\nCannot start — fix the hard failures above (or run `npc doctor`).",
              file=sys.stderr)
        return 1
    failed_soft = {c.name.split(" (")[0] for c in checks if not c.ok}
    if "Audio subsystem" in failed_soft:
        failed_soft |= {"Audio input", "Audio output"}

    llm = make_llm_client(config.llm)

    transcriber = recorder = None
    if not ({"Whisper model", "Audio input", "Push-to-talk"} & failed_soft):
        from .audio.recorder import PushToTalkRecorder, VadRecorder
        from .stt import WhisperTranscriber

        print("Loading whisper model…")
        transcriber = WhisperTranscriber(config.stt.model, config.stt.language,
                                         config.stt.device)
        if config.hotkey.mode == "tap":
            recorder = VadRecorder(threshold_db=config.stt.silence_threshold_db,
                                   silence_seconds=config.stt.vad_silence_seconds,
                                   max_seconds=config.stt.vad_max_seconds)
        else:
            recorder = PushToTalkRecorder()
    else:
        print("! voice input disabled this session (see FAILs above); /say still works")

    speaker = make_speaker = None
    if "Piper voice" not in failed_soft and "Audio output" not in failed_soft:
        from .audio.player import AudioPlayer
        from .tts import PiperSpeaker

        # ONE AudioPlayer for every voice: its persistent output stream is
        # what keeps the ALSA PipeWire plugin's close bug out of sessions
        player = AudioPlayer()
        speaker = PiperSpeaker(config.tts.voice_path, player=player)

        def make_speaker(voice_path, _player=player):
            return PiperSpeaker(voice_path, player=_player)
    else:
        print("! spoken replies disabled this session (see FAILs above)")

    overlay = None

    def on_event(event):
        print_event(event)
        if args.timings and isinstance(event, TurnCompleted):
            print(format_timings(event))
        if overlay is not None:
            if isinstance(event, NpcSwitched):
                # late-connecting OBS pages get the current name in Hello
                overlay.hello["npc_name"] = event.npc_name
            overlay.publish(event)

    app = NPCApp(config, llm=llm, transcriber=transcriber, recorder=recorder,
                 speaker=speaker, make_speaker=make_speaker, on_event=on_event)

    if config.overlay.enabled or args.overlay:
        from .overlay import OverlayServer

        try:
            overlay = OverlayServer(config.overlay.port,
                                    hello={"npc_name": app.npc_name,
                                           "session_no": app.session_no})
            overlay.start()
            print(f"overlay: http://127.0.0.1:{overlay.port}")
        except Exception as e:
            overlay = None
            print(f"! overlay disabled: {e}")

    app.start()
    print(f"\n{app.npc_name} is listening — session {app.session_no}.")
    if len(app.roster) > 1:
        print(f"{len(app.roster)} NPCs in this campaign — /npc lists and switches.")
    verb = "Tap" if config.hotkey.mode == "tap" else "Hold"
    print(f"{verb} {config.hotkey.key} to speak to the NPC. Type /help for commands.\n")

    action = _run_repl(app, config, ptt_enabled=recorder is not None)
    app.shutdown(summarize=(action == "end"))  # end-of-session events still broadcast
    if overlay is not None:
        overlay.stop()
    print("Farewell.")
    return 0


def _run_repl(app, config, ptt_enabled: bool) -> str:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.patch_stdout import patch_stdout

    session = PromptSession("gm> ")
    listener = None
    if ptt_enabled:
        try:
            listener = _start_hotkey(app, config, session)
        except Exception as e:
            print(f"! push-to-talk disabled: {e}")

    try:
        with patch_stdout():
            while True:
                try:
                    line = session.prompt()
                except (EOFError, KeyboardInterrupt):
                    return "quit"
                action = app.handle_line(line)
                if action != "ok":
                    return action
    finally:
        if listener is not None:
            listener.stop()


def _start_hotkey(app, config, session):
    """Wire evdev press/release to the app; see _ptt_callbacks for how the
    hotkey coexists with the REPL when it is an ordinary typing key."""
    from .hotkey import PTTListener, find_ptt_devices, keycode_from_name

    keycode = keycode_from_name(config.hotkey.key)
    devices = find_ptt_devices(keycode, config.hotkey.device)
    typing_key = _key_types_text(config.hotkey.key) and not config.hotkey.grab
    on_press, on_release = _ptt_callbacks(app, session, typing_key=typing_key,
                                          tap_mode=config.hotkey.mode == "tap")
    listener = PTTListener(devices, keycode, on_press, on_release,
                           grab=config.hotkey.grab)
    listener.start()
    return listener


def _key_types_text(key_name: str) -> bool:
    """True for hotkeys that also insert a character into the terminal."""
    import re

    return bool(re.fullmatch(
        r"KEY_(SPACE|TAB|[A-Z0-9]|MINUS|EQUAL|COMMA|DOT|SLASH|SEMICOLON|APOSTROPHE|GRAVE)",
        key_name))


def _ptt_callbacks(app, session, typing_key: bool, tap_mode: bool = False):
    """Press/release handlers that coexist with the REPL.

    A hotkey that is an ordinary typing key (the default spacebar) causes two
    terminal artifacts:
    - pressing it while a line is half-typed (an OOC note, /say …) is typing,
      not push-to-talk — ignore the press entirely and let the character land
      in the buffer;
    - holding it on an empty line types characters into the prompt — snapshot
      the buffer on press and restore it (and flush stdin) on release.

    In tap mode the key toggles instead of holding: first press starts the
    recording (silence ends it via the VAD recorder), a second press while
    RECORDING stops it early; key release only does the terminal cleanup.
    """
    import termios

    from .events import State

    state = {"snapshot": "", "typing": False}

    def buffer_text() -> str:
        try:
            return session.app.current_buffer.text
        except Exception:
            return ""

    def on_press():
        text = buffer_text()
        if typing_key and text.strip():
            state["typing"] = True
            return
        state["typing"] = False
        state["snapshot"] = text
        if tap_mode and app.state is State.RECORDING:
            app.on_ptt_release()  # second tap = stop early
        else:
            app.on_ptt_press()

    def on_release():
        if state["typing"]:
            state["typing"] = False
            return
        if not tap_mode:
            app.on_ptt_release()
        try:
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
            pt_app = session.app
            saved = state["snapshot"]

            def restore():
                from prompt_toolkit.document import Document

                pt_app.current_buffer.set_document(Document(saved, len(saved)))

            if pt_app.is_running and pt_app.loop is not None:
                pt_app.loop.call_soon_threadsafe(restore)
        except Exception:
            pass

    return on_press, on_release


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="npc",
        description="Offline AI NPC voice agent for tabletop RPGs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, func, help_text):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("campaign", help="campaign directory")
        p.set_defaults(func=func)
        return p

    add("init", cmd_init, "scaffold a new campaign directory")
    p = add("run", cmd_run, "run the NPC for a play session")
    p.add_argument("--timings", action="store_true",
                   help="print per-stage turn timings after each reply")
    p.add_argument("--overlay", action="store_true",
                   help="serve the OBS/table overlay (forces [overlay] enabled)")
    p = add("doctor", cmd_doctor, "check (and set up) everything the NPC needs")
    p.add_argument("--fix", action="store_true",
                   help="offer to run the safe fixes (model pull, voice download)")
    p = add("transcribe", cmd_transcribe, "debug: transcribe a wav file")
    p.add_argument("file", help="16-bit PCM wav file")
    p = add("say", cmd_say, "debug: speak a line with the NPC voice")
    p.add_argument("text", help="text to speak")

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
