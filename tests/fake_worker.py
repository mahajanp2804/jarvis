"""Stand-in model worker speaking the jarvis.ai.worker protocol (used by tests).

Usage: python fake_worker.py MODE   where MODE is ok | fail_load | crash | hang
"""

from __future__ import annotations

import json
import sys
import time


def main() -> int:
    mode = sys.argv[1]
    for line in sys.stdin:
        request = json.loads(line)
        op = request["op"]
        if op == "close":
            return 0
        if op == "load":
            if mode == "fail_load":
                reply = {
                    "ok": False,
                    "error_type": "ModelFileError",
                    "message": "Model file not found: x.gguf",
                    "hint": "download it",
                }
            else:
                reply = {"ok": True}
        elif op == "complete":
            if mode == "crash":
                return 3
            if mode == "hang":
                time.sleep(60)
            reply = {
                "ok": True,
                "completion": {
                    "text": json.dumps({"echo": request["messages"][-1]["content"]}),
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "seconds": 0.01,
                    "prompt_eval_seconds": None,
                    "generation_seconds": None,
                    "cached_prompt_tokens": None,
                },
            }
        else:
            reply = {"ok": False, "error_type": "ProtocolError", "message": "bad op"}
        print(json.dumps(reply), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
