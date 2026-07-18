"""OverlayServer: JSON serialization + a real localhost round-trip."""

import json

import httpx
import pytest

from npc.events import NpcReplied, NpcSwitched, State, StateChanged, TurnCompleted
from npc.overlay import OverlayServer, event_to_json


def test_event_to_json_types_and_enums():
    assert json.loads(event_to_json(StateChanged(State.SPEAKING))) == {
        "type": "StateChanged", "state": "speaking"}
    assert json.loads(event_to_json(NpcReplied("Vess", "Well met."))) == {
        "type": "NpcReplied", "npc_name": "Vess", "text": "Well met."}
    timings = json.loads(event_to_json(TurnCompleted(None, 0.3, 2.0, None, 4.5)))
    assert timings["stt_seconds"] is None
    assert timings["llm_first_token_seconds"] == 0.3
    assert json.loads(event_to_json(NpcSwitched("Korval the Red", None))) == {
        "type": "NpcSwitched", "npc_name": "Korval the Red", "voice": None}


@pytest.fixture
def server():
    overlay = OverlayServer(port=0, hello={"npc_name": "Vess", "session_no": 1})
    overlay.start()
    yield overlay
    overlay.stop()


def test_websocket_hello_and_broadcast(server):
    from websockets.sync.client import connect

    with connect(f"ws://127.0.0.1:{server.port}/ws") as ws:
        hello = json.loads(ws.recv(timeout=2))
        assert hello == {"type": "Hello", "npc_name": "Vess", "session_no": 1}
        server.publish(NpcReplied("Vess", "Well met."))  # from the test thread
        message = json.loads(ws.recv(timeout=2))
        assert message == {"type": "NpcReplied", "npc_name": "Vess",
                           "text": "Well met."}


def test_overlay_page_served_over_plain_http(server):
    response = httpx.get(f"http://127.0.0.1:{server.port}/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<!doctype html>" in response.text.lower()
    assert "WebSocket" in response.text


def test_publish_after_stop_is_a_silent_noop():
    overlay = OverlayServer(port=0)
    overlay.start()
    overlay.stop()
    overlay.publish(StateChanged(State.IDLE))  # must not raise


def test_port_collision_raises_at_start():
    first = OverlayServer(port=0)
    first.start()
    second = OverlayServer(port=first.port)
    with pytest.raises(OSError):
        second.start()
    first.stop()


def test_dm_only_events_are_marked_and_defaults_are_not():
    """cli.on_event skips overlay.publish for type(event).dm_only — these
    flags ARE the leak gate for secret ids/hints on the table websocket."""
    from npc.events import (
        Event,
        GmNoteAdded,
        Info,
        SecretList,
        SecretNote,
        SecretPending,
        SecretPondering,
        SecretResolved,
        SecretRevealRequested,
    )

    dm_only = (SecretRevealRequested, SecretResolved, SecretPending,
               SecretList, SecretNote)
    for cls in dm_only:
        assert cls.dm_only is True, cls.__name__
    for cls in (Event, StateChanged, NpcReplied, Info, GmNoteAdded,
                SecretPondering):  # pondering is table-safe on purpose
        assert cls.dm_only is False, cls.__name__


def test_secret_pondering_serializes_content_free():
    from npc.events import SecretPondering

    data = json.loads(event_to_json(SecretPondering("Vess", active=True)))
    assert data == {"type": "SecretPondering", "npc_name": "Vess", "active": True}


def test_event_json_keeps_utf8_readable():
    """Swedish text crosses the websocket as UTF-8, not \\u-escapes."""
    payload = event_to_json(NpcReplied("Vess", "Hertigen är begravd vid fyren…"))
    assert "Hertigen är begravd vid fyren…" in payload
    assert "\\u" not in payload
