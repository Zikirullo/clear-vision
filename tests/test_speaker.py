"""Tests for src/speaker.py queue/priority rules, using a silent fake backend."""

from __future__ import annotations

import threading
import time

import pytest

from src.navigator import Message
from src.speaker import Speaker

TIMEOUT = 2.0


class FakeBackend:
    """Records what was spoken; each utterance lasts until released or cancelled."""

    name = "fake"

    def __init__(self) -> None:
        self.started: list[str] = []
        self.finished: list[tuple[str, bool]] = []  # (text, was_interrupted)
        self._release = threading.Event()

    def speak(self, text: str, cancelled: threading.Event) -> None:
        self.started.append(text)
        while not cancelled.is_set() and not self._release.is_set():
            cancelled.wait(0.01)
        self._release.clear()
        self.finished.append((text, cancelled.is_set()))

    def interrupt(self) -> None:
        pass  # the cancelled event is what stops the fake

    def release(self) -> None:
        """Let the current utterance finish normally."""
        self._release.set()

    def wait_started(self, count: int) -> None:
        deadline = time.monotonic() + TIMEOUT
        while time.monotonic() < deadline:
            if len(self.started) >= count:
                return
            time.sleep(0.005)
        raise AssertionError(f"expected {count} utterances, got {self.started}")


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def speaker(backend: FakeBackend):
    with Speaker(backend=backend) as s:
        yield s


def finish_all(speaker: Speaker, backend: FakeBackend) -> None:
    """Let every queued utterance finish normally."""
    for _ in range(int(TIMEOUT / 0.05)):
        if speaker.wait_until_idle(0.05):
            return
        backend.release()
    raise AssertionError("speaker did not become idle")


def test_say_returns_immediately_and_speaks(speaker: Speaker, backend: FakeBackend) -> None:
    assert speaker.say("Person, left") is True
    backend.wait_started(1)
    finish_all(speaker, backend)
    assert backend.finished == [("Person, left", False)]


def test_urgent_interrupts_current_speech(speaker: Speaker, backend: FakeBackend) -> None:
    speaker.say("A long normal sentence")
    backend.wait_started(1)
    speaker.say("Car ahead, close", priority="urgent")
    backend.wait_started(2)
    finish_all(speaker, backend)
    assert backend.finished == [("A long normal sentence", True), ("Car ahead, close", False)]


def test_normal_does_not_interrupt(speaker: Speaker, backend: FakeBackend) -> None:
    speaker.say("First")
    backend.wait_started(1)
    speaker.say("Second")
    finish_all(speaker, backend)
    assert backend.finished == [("First", False), ("Second", False)]


def test_only_newest_pending_normal_is_kept(speaker: Speaker, backend: FakeBackend) -> None:
    speaker.say("Speaking now")
    backend.wait_started(1)
    speaker.say("Old pending")
    speaker.say("New pending")
    finish_all(speaker, backend)
    assert backend.started == ["Speaking now", "New pending"]


def test_normal_dropped_when_urgent_pending(speaker: Speaker, backend: FakeBackend) -> None:
    # Hold the worker on an urgent utterance so the next urgent stays pending.
    speaker.say("Urgent one", priority="urgent")
    backend.wait_started(1)
    with speaker._cond:  # queue two messages atomically, before the worker reacts
        speaker._pending = Message("Urgent two", "urgent", "u2")
    assert speaker.say("Normal message") is False
    finish_all(speaker, backend)
    assert "Normal message" not in backend.started


def test_urgent_replaces_pending_normal(speaker: Speaker, backend: FakeBackend) -> None:
    speaker.say("Speaking now")
    backend.wait_started(1)
    with speaker._cond:
        speaker._pending = Message("Pending normal", "normal", "n")
    speaker.say("Urgent", priority="urgent")
    finish_all(speaker, backend)
    assert backend.started == ["Speaking now", "Urgent"]


def test_accepts_navigator_message(speaker: Speaker, backend: FakeBackend) -> None:
    speaker.say(Message("Car ahead, close", "urgent", "car:ahead"))
    backend.wait_started(1)
    finish_all(speaker, backend)
    assert backend.started == ["Car ahead, close"]


def test_stop_cuts_off_speech(backend: FakeBackend) -> None:
    speaker = Speaker(backend=backend).start()
    speaker.say("Long sentence")
    backend.wait_started(1)
    speaker.stop()
    assert backend.finished == [("Long sentence", True)]
