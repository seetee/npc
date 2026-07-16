"""OverlayServer: JSON serialization + a real localhost round-trip."""

import json

import httpx
import pytest

from npc.events import NpcReplied, State, StateChanged, TurnCompleted
from npc.overlay import OverlayServer, event_to_json


def test_event_to_json_types_and_enums():
    assert json.loads(event_to_json(StateChanged(State.SPEAKING))) == {
        "type": "StateChanged", "state": "speaking"}
    assert json.loads(event_to_json(NpcReplied("Vess", "Well met."))) == {
        "type": "NpcReplied", "npc_name": "Vess", "text": "Well met."}
    timings = json.loads(event_to_json(TurnCompleted(None, 0.3, 2.0, None, 4.5)))
    assert timings["stt_seconds"] is None
    assert timings["llm_first_token_seconds"] == 0.3


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
