from datetime import datetime

from npc.session.logbook import Logbook, Transcript


def test_session_numbering_starts_at_one(tmp_path):
    book = Logbook(tmp_path / "logbook.md")
    assert book.next_session_number() == 1


def test_upsert_appends_then_replaces(tmp_path):
    book = Logbook(tmp_path / "logbook.md")
    book.upsert_entry(1, "2026-07-15", "**Location:** the docks")
    book.upsert_entry(2, "2026-07-22", "**Location:** the ruins")
    assert book.next_session_number() == 3

    # checkpoint re-summarizes session 2: replaced, not duplicated
    book.upsert_entry(2, "2026-07-22", "**Location:** deep in the ruins")
    text = (tmp_path / "logbook.md").read_text()
    assert text.count("## Session 2") == 1
    assert "deep in the ruins" in text
    assert "## Session 1" in text  # untouched


def test_tail_returns_last_sections(tmp_path):
    book = Logbook(tmp_path / "logbook.md")
    for n in range(1, 5):
        book.upsert_entry(n, f"2026-07-{n:02d}", f"summary {n}")
    tail = book.tail(2)
    assert "summary 3" in tail and "summary 4" in tail
    assert "summary 2" not in tail
    assert book.tail(99).count("## Session") == 4


def test_tail_of_empty_logbook(tmp_path):
    assert Logbook(tmp_path / "logbook.md").tail(3) == ""


def test_transcript_appends_turns(tmp_path):
    transcript = Transcript(tmp_path / "sessions",
                            now=datetime(2026, 7, 15, 19, 30))
    transcript.append_turn("PLAYER", "who are you?")
    transcript.append_turn("NPC", "I am Vess.")
    assert transcript.path.name == "2026-07-15-1930-transcript.md"
    content = transcript.read()
    assert "**PLAYER:** who are you?" in content
    assert "**NPC:** I am Vess." in content
