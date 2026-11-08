"""Non-blocking text-to-speech with priorities.

`Speaker.say()` returns immediately; a worker thread does the speaking, so the
camera loop never waits on audio. There is at most one pending message:

- urgent: replaces whatever is pending and interrupts current speech;
- normal: replaces an older pending normal message (newest wins), but is
  dropped if an urgent message is pending. It never interrupts speech.

Backends: pyttsx3, or the macOS `say` command. On macOS, "auto" uses `say`:
pyttsx3's macOS driver returns from runAndWait() immediately when called off
the main thread, so the end of an utterance can't be detected and messages
would overlap. If pyttsx3 raises at runtime, the speaker falls back to `say`.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, Protocol

from src.navigator import Message, Priority

logger = logging.getLogger(__name__)

DEFAULT_RATE = 220
BackendName = Literal["auto", "pyttsx3", "say"]


class SpeechBackend(Protocol):
    name: str

    def speak(self, text: str, cancelled: threading.Event) -> None:
        """Speak `text`, blocking until done or until `cancelled` is set."""

    def interrupt(self) -> None:
        """Stop current speech as soon as possible (called from another thread)."""


class SayBackend:
    """macOS `say` command; interrupted by terminating the process."""

    name = "say"

    def __init__(self, rate: int = DEFAULT_RATE) -> None:
        if shutil.which("say") is None:
            raise RuntimeError("macOS `say` command not found")
        self.rate = rate
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None

    def speak(self, text: str, cancelled: threading.Event) -> None:
        proc = subprocess.Popen(["say", "-r", str(self.rate), text])
        with self._lock:
            self._proc = proc
            # interrupt() may have run before the process existed.
            if cancelled.is_set():
                proc.terminate()
        proc.wait()
        with self._lock:
            self._proc = None

    def interrupt(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()


class Pyttsx3Backend:
    """pyttsx3 engine, created lazily on the worker thread that uses it."""

    name = "pyttsx3"

    def __init__(self, rate: int = DEFAULT_RATE) -> None:
        import pyttsx3  # noqa: F401  (fail early if it isn't installed)

        self.rate = rate
        self._engine: Any = None

    def speak(self, text: str, cancelled: threading.Event) -> None:
        if self._engine is None:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
        if cancelled.is_set():
            return
        self._engine.say(text)
        self._engine.runAndWait()

    def interrupt(self) -> None:
        if self._engine is not None:
            self._engine.stop()


def make_backend(name: BackendName, rate: int) -> SpeechBackend:
    if name == "say" or (name == "auto" and sys.platform == "darwin"):
        return SayBackend(rate)
    try:
        return Pyttsx3Backend(rate)
    except Exception:
        if sys.platform == "darwin":
            logger.exception("pyttsx3 unavailable; falling back to `say`")
            return SayBackend(rate)
        raise


@dataclass
class _Utterance:
    message: Message
    cancelled: threading.Event


class Speaker:
    """Speaks messages on a background thread. See module docstring for rules.

    Args:
        rate: Speech rate in words per minute.
        backend: "auto", "pyttsx3", "say", or a backend instance (for tests).
    """

    def __init__(self, rate: int = DEFAULT_RATE, backend: BackendName | SpeechBackend = "auto") -> None:
        self.rate = rate
        self._backend: SpeechBackend = (
            make_backend(backend, rate) if isinstance(backend, str) else backend
        )
        self._cond = threading.Condition()
        self._pending: Message | None = None
        self._current: _Utterance | None = None
        self._stopping = False
        self._thread: threading.Thread | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Speaker:
        """Build from the `speaker:` section of configs/default.yaml."""
        return cls(rate=int(cfg.get("rate", DEFAULT_RATE)), backend=cfg.get("backend", "auto"))

    @property
    def backend_name(self) -> str:
        return self._backend.name

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> Speaker:
        if self._thread is None or not self._thread.is_alive():
            self._stopping = False
            self._thread = threading.Thread(target=self._run, name="speaker", daemon=True)
            self._thread.start()
            logger.info("Speaker started: backend=%s rate=%d", self._backend.name, self.rate)
        return self

    def stop(self) -> None:
        """Stop immediately, cutting off current speech and dropping pending."""
        with self._cond:
            self._stopping = True
            self._pending = None
            if self._current is not None:
                self._current.cancelled.set()
            self._cond.notify_all()
        self._backend.interrupt()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> Speaker:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    # ------------------------------------------------------------------ public API

    def say(self, message: Message | str, priority: Priority = "normal") -> bool:
        """Queue a message without blocking. Returns False if it was dropped.

        A plain string is wrapped in a Message with the given priority.
        """
        if isinstance(message, str):
            message = Message(text=message, priority=priority, key=message)

        interrupt = False
        with self._cond:
            if message.priority == "urgent":
                if self._pending is not None:
                    logger.debug("Urgent replaces pending: %r", self._pending.text)
                self._pending = message
                if self._current is not None:
                    self._current.cancelled.set()
                    interrupt = True
            else:
                if self._pending is not None and self._pending.priority == "urgent":
                    logger.debug("Dropped (urgent pending): %r", message.text)
                    return False
                if self._pending is not None:
                    logger.debug("Newer message replaces pending: %r", self._pending.text)
                self._pending = message
            self._cond.notify_all()

        if interrupt:
            logger.info("Interrupting speech for urgent: %r", message.text)
            self._backend.interrupt()
        return True

    @property
    def is_speaking(self) -> bool:
        with self._cond:
            return self._current is not None

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        """Block until nothing is speaking or pending (for scripts and tests)."""
        with self._cond:
            return self._cond.wait_for(
                lambda: self._current is None and self._pending is None, timeout
            )

    # ------------------------------------------------------------------ worker

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._pending is not None or self._stopping)
                if self._stopping:
                    return
                assert self._pending is not None
                utterance = _Utterance(self._pending, threading.Event())
                self._pending = None
                self._current = utterance

            logger.info("Speaking [%s]: %s", utterance.message.priority, utterance.message.text)
            try:
                self._backend.speak(utterance.message.text, utterance.cancelled)
            except Exception:
                self._handle_backend_failure(utterance)

            with self._cond:
                self._current = None
                self._cond.notify_all()

    def _handle_backend_failure(self, utterance: _Utterance) -> None:
        if self._backend.name == "pyttsx3" and sys.platform == "darwin":
            logger.exception("pyttsx3 failed; switching to `say`")
            self._backend = SayBackend(self.rate)
            if not utterance.cancelled.is_set():
                self._backend.speak(utterance.message.text, utterance.cancelled)
        else:
            # Speech is the only output a blind user gets, so this is critical.
            logger.critical("Speech backend %s failed", self._backend.name, exc_info=True)
