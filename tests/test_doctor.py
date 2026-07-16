"""doctor --fix: the interactive fixer loop (no network, no hardware)."""

import pytest

from npc.doctor import CheckResult, apply_fixes


def test_apply_fixes_runs_only_confirmed_fixers():
    ran = []
    checks = [
        CheckResult("passing", True, fixer=lambda: ran.append("passing")),
        CheckResult("no-fixer", False, fix="sudo something  # print-only"),
        CheckResult("confirmed", False, fixer=lambda: ran.append("confirmed"),
                    fix_label="download the thing"),
        CheckResult("declined", False, fixer=lambda: ran.append("declined")),
    ]
    answers = iter(["y", "n"])
    out = []
    assert apply_fixes(checks, ask=lambda prompt: next(answers), out=out.append)
    assert ran == ["confirmed"]
    assert any("fixed: confirmed" in line for line in out)


def test_apply_fixes_survives_a_raising_fixer():
    def boom():
        raise RuntimeError("download failed")

    out = []
    assert apply_fixes([CheckResult("bad", False, fixer=boom, fix_label="x")],
                       ask=lambda prompt: "y", out=out.append)
    assert any("fix failed for bad" in line for line in out)


def test_apply_fixes_asks_nothing_when_all_pass():
    checks = [CheckResult("fine", True, fixer=lambda: None)]
    assert not apply_fixes(checks, ask=lambda prompt: pytest.fail("should not ask"),
                           out=lambda line: None)
