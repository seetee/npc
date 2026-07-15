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


def test_body_with_regex_escapes_survives_verbatim(tmp_path):
    """LLM output may contain backslashes / group refs; upsert must treat the
    replacement as literal text, both on first write and on re-summarize."""
    book = Logbook(tmp_path / "logbook.md")
    body = r"paid 10\1 shins, \g<0> and a \\backslash"
    book.upsert_entry(1, "2026-07-15", body)
    book.upsert_entry(1, "2026-07-15", body)  # replace path uses re.sub
    assert body in (tmp_path / "logbook.md").read_text()


def test_session_headings_in_body_are_demoted(tmp_path):
    """A '## Session N' line inside an LLM summary must not corrupt parsing."""
    book = Logbook(tmp_path / "logbook.md")
    book.upsert_entry(1, "2026-07-15", "**Highlights:**\n## Session 99 — echoed\nstuff")
    assert book.next_session_number() == 2          # 99 not parsed as a session
    assert "### Session 99" in book.tail(1)          # demoted, content kept
    book.upsert_entry(2, "2026-07-22", "later")
    assert "stuff" not in book.tail(1)               # sections still split right


def test_upsert_leaves_no_temp_files(tmp_path):
    book = Logbook(tmp_path / "logbook.md")
    book.upsert_entry(1, "2026-07-15", "body")
    leftovers = [p for p in tmp_path.iterdir() if p.name != "logbook.md"]
    assert leftovers == []


def test_transcript_appends_turns(tmp_path):
    transcript = Transcript(tmp_path / "sessions",
                            now=datetime(2026, 7, 15, 19, 30))
    transcript.append_turn("PLAYER", "who are you?")
    transcript.append_turn("NPC", "I am Vess.")
    assert transcript.path.name == "2026-07-15-1930-transcript.md"
    content = transcript.read()
    assert "**PLAYER:** who are you?" in content
    assert "**NPC:** I am Vess." in content
