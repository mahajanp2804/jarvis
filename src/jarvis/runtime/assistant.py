"""Activation flow for the background service (GUI-agnostic).

    shortcut -> prompt (main thread) -> plan + execute (worker thread)
             -> clarifying question? ask, continue -> show reply if useful -> idle

With spoken replies on, a short summary is said as the reply appears;
pressing the shortcut again stops it.

Push-to-talk (its own shortcut):

    shortcut -> "Listening…" (worker process records) -> pause, Done or shortcut again
             -> transcribe (worker thread) -> editable transcript -> Run -> as above

Nothing is recorded before the shortcut is pressed, and the transcript only
runs after the user has seen it and pressed Run.

Requests run on a single worker thread, so the main thread stays free for the
run loop, dialogs and signals. Only one request runs at a time. Confirmations
requested by the dispatcher are shown on the main thread through
:class:`DialogConfirmer`, and default to "no" if Jarvis stops while asking.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from jarvis.agent.orchestrator import Orchestrator, Reply
from jarvis.agent.planner import PlanAction
from jarvis.api.schemas import ToolStatus
from jarvis.errors import JarvisError
from jarvis.runtime.event_loop import EventLoop
from jarvis.ui.base import UserInterface
from jarvis.voice.listen import Recording, VoiceInput
from jarvis.voice.speech import DEFAULT_SPOKEN_CHARS, SpeechOutput, spoken_text

log = logging.getLogger(__name__)

TITLE = "Jarvis"
# Tools whose success is visible on screen, so no dialog is needed afterwards.
QUIET_TOOLS = frozenset({"apps.open", "apps.quit"})


class DialogConfirmer:
    """:class:`~jarvis.runtime.permissions.Confirmer` that asks with a dialog."""

    def __init__(self, loop: EventLoop, ui: UserInterface) -> None:
        self._loop = loop
        self._ui = ui

    def confirm(self, prompt: str) -> bool:
        def ask() -> bool:
            return self._ui.confirm("Jarvis needs your permission", prompt)

        if threading.current_thread() is threading.main_thread():
            return ask()
        return self._loop.call_and_wait(ask, default=False)


def needs_answer(reply: Reply) -> bool:
    """True when Jarvis asked the user a question."""
    if reply.plan is not None and reply.plan.action is PlanAction.CLARIFY:
        return True
    return any(r.status is ToolStatus.NEEDS_CLARIFICATION for r in reply.results)


def should_show(reply: Reply, *, show_success: bool = False) -> bool:
    """Whether a reply deserves a dialog. Visible successes and declines don't."""
    if not reply.results:
        return True  # text answers, errors, "not understood"
    statuses = {r.status for r in reply.results}
    if statuses == {ToolStatus.CANCELLED}:
        return False  # the user just said no
    visible_success = all(
        r.status is ToolStatus.SUCCESS and r.verified is True and r.tool in QUIET_TOOLS
        for r in reply.results
    )
    return show_success or not visible_success


class Assistant:
    """Runs requests from the shortcut or the control socket."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        ui: UserInterface,
        loop: EventLoop,
        *,
        show_success: bool = False,
        max_follow_ups: int = 3,
        speaker: SpeechOutput | None = None,
        spoken_chars: int = DEFAULT_SPOKEN_CHARS,
        voice: VoiceInput | None = None,
        voice_shortcut: str = "the shortcut",
    ) -> None:
        self._orchestrator = orchestrator
        self._speaker = speaker
        self._spoken_chars = spoken_chars
        self._voice = voice
        self._voice_shortcut = voice_shortcut
        self._recording: Recording | None = None  # only touched on the main thread
        self._ui = ui
        self._loop = loop
        self._show_success = show_success
        self._max_follow_ups = max_follow_ups
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-task")
        self._lock = threading.Lock()
        self._current: Future[Reply] | None = None
        self.requests_handled = 0

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._current is not None and not self._current.done()

    def activate(self) -> None:
        """Shortcut pressed (runs on the main thread)."""
        if self._loop.stopping:
            return
        if self._speaker is not None:
            self._speaker.stop()  # the shortcut interrupts a spoken reply
        if self._recording is not None:
            return  # the listening dialog is open
        if self.busy:
            self._ui.show(TITLE, "Still working on the previous request…")
            return
        text = self._ui.prompt(TITLE, "What should I do?")
        if text and not self._loop.stopping:
            self.submit(text, interactive=True)

    def push_to_talk(self) -> None:
        """Push-to-talk shortcut pressed (main thread): start, or finish, a recording."""
        if self._loop.stopping or self._voice is None:
            return
        if self._speaker is not None:
            self._speaker.stop()
        if self._recording is not None:
            self._recording.stop()  # pressed again: done talking
            return
        if self.busy:
            self._ui.show(TITLE, "Still working on the previous request…")
            return
        try:
            recording = self._voice.start()
        except JarvisError as exc:
            self._ui.show(TITLE, _explain(exc))
            return
        self._recording = recording
        try:
            use = self._ui.listen(
                TITLE,
                f"Listening… Say your request, then pause (or press {self._voice_shortcut} "
                "again, or click Done).",
                recording.recorded,
            )
        finally:
            self._recording = None
        if not use or self._loop.stopping:
            recording.cancel()
            return
        recording.stop()
        future = self._executor.submit(self._transcribe_and_run, recording)
        with self._lock:
            self._current = future

    def submit(self, text: str, *, interactive: bool) -> Future[Reply]:
        """Queue a request. ``interactive`` requests may ask follow-ups and show replies."""
        future = self._executor.submit(self._run, text, interactive)
        with self._lock:
            self._current = future
        return future

    def shutdown(self, timeout: float) -> bool:
        """Stop accepting work and wait for the running request. True if it finished."""
        with self._lock:
            current = self._current
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._speaker is not None:
            self._speaker.stop()
        if self._recording is not None:
            self._recording.cancel()
        if current is None:
            return True
        try:
            current.result(timeout=timeout)
        except FutureTimeout:
            log.warning("Request still running after %.0fs at shutdown", timeout)
            return False
        except Exception:
            log.debug("Request ended with an error during shutdown", exc_info=True)
        return True

    # -- worker thread ---------------------------------------------------------------

    def _say(self, reply: Reply) -> None:
        if self._speaker is None:
            return
        try:
            self._speaker.speak(spoken_text(reply, self._spoken_chars))
        except JarvisError as exc:
            log.warning("Spoken reply failed: %s", exc.message)

    def _show(self, message: str) -> None:
        self._loop.call_and_wait(lambda: self._ui.show(TITLE, message), default=None)

    def _transcribe_and_run(self, recording: Recording) -> Reply:
        try:
            transcript = recording.result()
        except JarvisError as exc:
            self._show(_explain(exc))
            return Reply(exc.message, handled=False)
        if not transcript.text:
            self._show("I didn't hear anything.")
            return Reply("I didn't hear anything.", handled=False)
        heard = transcript.text

        def edit() -> str | None:
            return self._ui.edit(TITLE, "I heard this. Fix anything that's wrong, then Run:", heard)

        text = self._loop.call_and_wait(edit, default=None)
        if not text or self._loop.stopping:
            return Reply("Cancelled.", handled=False)
        return self._run(text, interactive=True, source="voice")

    def _ask(self, question: str) -> str | None:
        def prompt() -> str | None:
            return self._ui.prompt(TITLE, question)

        return self._loop.call_and_wait(prompt, default=None)

    def _run(self, text: str, interactive: bool, source: str = "service") -> Reply:
        try:
            reply = self._orchestrator.handle(text, source=source)
            follow_ups = 0
            while (
                interactive
                and needs_answer(reply)
                and follow_ups < self._max_follow_ups
                and not self._loop.stopping
            ):
                self._say(reply)
                answer = self._ask(reply.text)
                if not answer:
                    break
                reply = self._orchestrator.handle(answer, source=source)
                follow_ups += 1
            if interactive and not self._loop.stopping:
                self._say(reply)
            if (
                interactive
                and not self._loop.stopping
                and should_show(reply, show_success=self._show_success)
            ):
                self._show(reply.text)
            return reply
        except Exception:
            log.exception("Request failed: %r", text)
            raise
        finally:
            self.requests_handled += 1


def _explain(exc: JarvisError) -> str:
    return exc.message + (f"\n\n{exc.hint}" if exc.hint else "")
