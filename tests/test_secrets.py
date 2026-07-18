"""Secrets file parsing, round-trip write-back, and marker detection."""

import pytest

from npc.session.secrets import (
    Secret,
    SecretsError,
    SecretsSheet,
    delivery_instruction,
    deny_note,
    find_markers,
    locked_block,
    revealed_block,
)

SAMPLE = """\
# Secrets — Elandra

Notes for the GM live up here and survive rewrites.

## duke-tomb
hint: where Duke Maren is really buried — share only if they earn her trust

The Duke lies in the sea-cave beneath the old lighthouse, not the family crypt.

## smuggler-name
hint: who runs the night shipments
mode: deflect
revealed: session 2

It is Alderman Voss himself.
"""


def test_parse_sample():
    sheet = SecretsSheet.parse(SAMPLE)
    assert "Notes for the GM" in sheet.preamble
    a, b = sheet.entries
    assert a.id == "duke-tomb" and a.mode == "hesitate" and a.revealed is None
    assert a.hint.startswith("where Duke Maren")
    assert "sea-cave" in a.body
    assert b.mode == "deflect" and b.revealed == "session 2"
    assert sheet.locked() == [a]
    assert sheet.revealed() == [b]
    assert sheet.get("smuggler-name") is b
    assert sheet.get("nope") is None


def test_render_round_trips():
    sheet = SecretsSheet.parse(SAMPLE)
    again = SecretsSheet.parse(sheet.render())
    assert again == sheet


def test_reveal_write_back(tmp_path):
    path = tmp_path / "secrets.md"
    path.write_text(SAMPLE, encoding="utf-8")
    sheet = SecretsSheet.load(path)
    sheet.get("duke-tomb").revealed = "session 3"
    sheet.save(path)
    reread = SecretsSheet.load(path)
    assert reread.get("duke-tomb").revealed == "session 3"
    assert reread.locked() == []
    assert "Notes for the GM" in reread.preamble


def test_load_missing_file_is_empty(tmp_path):
    sheet = SecretsSheet.load(tmp_path / "absent.md")
    assert sheet.entries == [] and sheet.locked() == []


def test_heading_id_is_normalized_lowercase():
    sheet = SecretsSheet.parse("## Duke-Tomb\nhint: h\n\nbody\n")
    assert sheet.entries[0].id == "duke-tomb"


@pytest.mark.parametrize("text, message", [
    ("## bad id\nhint: h\n\nbody\n", "not a valid secret id"),
    ("## a\n\nbody only\n", "missing its 'hint:'"),
    ("## a\nhint: h\nmode: shout\n\nbody\n", "mode 'shout'"),
    ("## a\nhint: h\n", "no body"),
    ("## a\nhint: h\n\nb\n\n## a\nhint: h\n\nb\n", "duplicate secret id"),
])
def test_parse_errors(text, message):
    with pytest.raises(SecretsError, match=message):
        SecretsSheet.parse(text)


def test_find_markers():
    raw = ("Hm. Let me think. [CHECK:duke-tomb] and again [check: Duke-Tomb ] "
           "plus [CHECK:other-one]")
    assert find_markers(raw) == ["duke-tomb", "other-one"]
    assert find_markers("no markers here") == []
    assert find_markers("[CHECK:]") == []


def secret(**kw):
    base = dict(id="duke-tomb", hint="where the Duke is buried",
                body="in the salt vault")
    base.update(kw)
    return Secret(**base)


def test_locked_block_lists_hints_never_bodies():
    block = locked_block([secret(), secret(id="smuggler-name", mode="deflect",
                                          hint="who runs the shipments",
                                          body="Alderman Voss")])
    assert "- topic (duke-tomb): where the Duke is buried" in block
    assert "  handling: hesitate\n  marker: [CHECK:duke-tomb]" in block
    assert "- topic (smuggler-name): who runs the shipments" in block
    assert "  handling: deflect\n  marker: [CHECK:smuggler-name]" in block
    assert "salt vault" not in block and "Alderman Voss" not in block


def test_revealed_block_contains_body():
    block = revealed_block([secret(revealed="session 1")])
    assert "## duke-tomb" in block and "salt vault" in block


def test_delivery_instruction_and_deny_note():
    s = secret()
    tell = delivery_instruction(s, rider="only vaguely, she is scared")
    assert "salt vault" in tell and "GM adds: only vaguely" in tell
    assert delivery_instruction(s).endswith("beyond it.")
    deny = deny_note(s, rider="she lies about the crypt")
    assert "About where the Duke is buried" in deny
    assert "GM adds: she lies about the crypt" in deny
    assert "salt vault" not in deny
    # the id must NOT appear: a 7B model rebuilds the marker from it
    assert "duke-tomb" not in deny and "marker" not in deny


def test_marker_scrubber_streams_clean_text():
    from npc.session.secrets import MarkerScrubber

    s = MarkerScrubber()
    out = s.feed("Hello there. [CHE") + s.feed("CK:duke-tomb] More text.")
    out += s.flush()
    assert out == "Hello there.  More text."

    s = MarkerScrubber()  # an unclosed non-marker bracket is released at flush
    assert s.feed("I saw [something") == "I saw "
    assert s.flush() == "[something"

    s = MarkerScrubber()  # whole marker in one chunk
    assert s.feed("Wait. [CHECK:x] Done.") == "Wait.  Done."
    assert s.flush() == ""
