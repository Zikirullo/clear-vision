"""Demonstrate Speaker priorities out loud.

1. A long normal sentence is cut off by an urgent warning.
2. Normal messages queued while speaking: only the newest is kept.
3. A normal message queued behind an urgent one is dropped.

Usage:
    python scripts/speaker_test.py
    python scripts/speaker_test.py --backend pyttsx3 --rate 180
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.speaker import Speaker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=["auto", "pyttsx3", "say"], help="override config")
    parser.add_argument("--rate", type=int, help="words per minute, overrides config")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.DEBUG, format="%(relativeCreated)6.0fms %(levelname)-5s %(message)s")
    args = parse_args()
    with (ROOT / "configs" / "default.yaml").open() as f:
        cfg = yaml.safe_load(f).get("speaker", {})
    if args.backend:
        cfg["backend"] = args.backend
    if args.rate:
        cfg["rate"] = args.rate

    with Speaker.from_config(cfg) as speaker:
        print(f"\nBackend: {speaker.backend_name}, rate {speaker.rate} wpm\n")

        print("1) Urgent interrupts a normal sentence")
        speaker.say("This is a long normal sentence about the weather, the street, "
                    "and the shops around you, and it should be cut off.")
        time.sleep(2.0)
        speaker.say("Stop! Car ahead, close", priority="urgent")
        speaker.wait_until_idle()
        time.sleep(1.0)

        print("\n2) Only the newest normal message is kept")
        speaker.say("Person, left")
        time.sleep(0.3)
        speaker.say("Bicycle, right")   # replaced by the next one before it is spoken
        speaker.say("Bench ahead")      # spoken after "Person, left"
        speaker.wait_until_idle()
        time.sleep(1.0)

        print("\n3) Normal is dropped while an urgent one is waiting")
        speaker.say("Chair, right")
        time.sleep(0.3)
        speaker.say("Person ahead, close", priority="urgent")
        dropped = not speaker.say("Dog, left")
        print(f"   'Dog, left' dropped: {dropped}")
        speaker.wait_until_idle()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
