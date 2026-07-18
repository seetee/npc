"""Drive a real scripted session for the README demo GIF.

    asciinema rec --overwrite -c "uv run python scripts/record_demo.py" demo.cast
    agg --cols 100 --rows 32 demo.cast docs/demo.gif

Runs `npc run examples/rusty-lantern` in a 100x30 pty (real Ollama, Piper
speaking out loud), types the script below with human-ish pacing, and
mirrors everything to stdout for asciinema to record. Re-record after UI
changes. Not part of the test suite.
"""

import fcntl
import os
import pty
import struct
import subprocess
import sys
import termios
import threading
import time

CAMPAIGN = "examples/rusty-lantern"
COLS, ROWS = 100, 30

# (wait_for_this_text, then_type_this) — waits scan the output stream in order
SCRIPT = [
    ("is listening", None),
    (None, "/say Good evening — are you Mera?"),
    ("[Mera Vex]", None),
    (None, "/say Folk say old Dannic did not die of marsh fever. What killed him?"),
    ("stays pending until you decide", None),
    (None, "/yes but only part of it — she is testing them"),
    ("[Mera Vex]", None),
    (None, "/secrets"),
    ("Secrets of Mera Vex", None),
    (None, "/quit"),
    ("Farewell", None),
]


def type_line(fd: int, line: str) -> None:
    time.sleep(1.2)  # a beat before "the GM starts typing"
    for ch in line:
        os.write(fd, ch.encode())
        time.sleep(0.045)
    time.sleep(0.4)
    os.write(fd, b"\r")


def main() -> int:
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    env = dict(os.environ, PROMPT_TOOLKIT_NO_CPR="1")  # no CPR warning on the pty
    proc = subprocess.Popen(
        ["uv", "run", "npc", "run", CAMPAIGN],
        stdin=slave, stdout=slave, stderr=slave, close_fds=True, env=env)
    os.close(slave)

    buffer = bytearray()
    scanned = 0  # matches consume the buffer in order

    def reader() -> None:
        # mirrors CONTINUOUSLY — asciinema must see typing echo and streamed
        # replies the moment they render, not in bursts between waits
        while True:
            try:
                data = os.read(master, 1024)
            except OSError:
                return
            if not data:
                return
            buffer.extend(data)
            os.write(1, data)

    threading.Thread(target=reader, daemon=True).start()

    def wait_for(text: str, timeout: float = 180.0) -> None:
        nonlocal scanned
        deadline = time.monotonic() + timeout
        needle = text.encode()
        while time.monotonic() < deadline:
            index = buffer.find(needle, scanned)
            if index != -1:
                scanned = index + len(needle)
                return
            time.sleep(0.05)
        raise SystemExit(f"timed out waiting for {text!r}")

    try:
        for wait, line in SCRIPT:
            if wait is not None:
                wait_for(wait)
            if line is not None:
                type_line(master, line)
        time.sleep(1.5)  # let the closing lines land in the recording
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        # the session was REAL: /yes wrote `revealed:` into the example's
        # secrets.md and a transcript landed in sessions/ — restore pristine
        subprocess.run(["git", "checkout", "--", CAMPAIGN], check=False)
        subprocess.run(["rm", "-rf", f"{CAMPAIGN}/sessions",
                        f"{CAMPAIGN}/logbook.md"], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
