"""Stand-in for jarvis.voice.stt_worker: emits scripted events (no microphone, no model).

Usage: python fake_voice_worker.py SCENARIO '<settings JSON>'
"""

from __future__ import annotations

import json
import sys
import threading


def emit(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}), flush=True)


def main() -> int:
    scenario = sys.argv[1]
    settings = json.loads(sys.argv[2])
    if scenario == "crash":
        return 1
    stopped = threading.Event()

    def watch() -> None:
        for line in sys.stdin:
            if line.strip() == "stop":
                break
        stopped.set()

    threading.Thread(target=watch, daemon=True).start()
    emit("listening")
    if scenario == "silent":
        emit("recorded", seconds=1.0, peak=0.0, speech=False, reason="no_speech")
        emit("error", kind="silent", message="The microphone gave only silence.", hint="grant it")
        return 4
    if scenario == "nothing":
        emit("recorded", seconds=8.0, peak=0.002, speech=False, reason="no_speech")
        emit("text", text="", seconds=0.0)
        return 0
    # "ok": wait for "stop" (or EOF), like a user who finishes talking.
    reason = "stop" if stopped.wait(timeout=float(settings.get("max_seconds", 5))) else "limit"
    emit("recorded", seconds=2.0, peak=0.4, speech=True, reason=reason)
    emit("text", text="open safari", seconds=0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
