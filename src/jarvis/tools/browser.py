"""Browser tools (Playwright): open, search, read, click, type, close.

Design and safety:

* **Its own profile.** Jarvis drives an installed Chrome (or Brave/Edge, or
  Playwright's Chromium) with a separate profile under Application Support, so
  your personal browser profile is never touched. Logins you do yourself in
  that window persist there.
* **DOM, not coordinates.** Pages are read as text plus a numbered list of
  interactive elements (``e1``, ``e2`` ...). Clicks and typing target those
  references, never screen positions.
* **Risky actions always ask.** Clicking anything that submits a form or reads
  like buy / pay / send / delete / sign up is HIGH risk. Typing and pressing
  Enter to submit is HIGH. Every page read is untrusted, so any later action
  in the same request needs confirmation anyway (see dispatcher taint).
* **Never bypasses security.** Jarvis won't type into password or payment
  fields, and stops at CAPTCHAs or login walls and asks you to handle them.
* **One browser thread.** Playwright's sync API is bound to the thread that
  created it, so all browser work runs on a dedicated thread. The browser
  starts on first use and closes after ``idle_close_seconds``.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import re
import threading
import urllib.parse
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from jarvis.api.schemas import RiskLevel, ToolResult, ToolStatus
from jarvis.config import BrowserConfig
from jarvis.errors import PolicyDeniedError, ToolExecutionError, ToolInputError
from jarvis.tools.base import PreparedAction, ToolSpec

log = logging.getLogger(__name__)
T = TypeVar("T")

ALLOWED_SCHEMES = ("http", "https")
_SENSITIVE = re.compile(
    r"(?i)\b(buy|purchase|pay|checkout|check out|place order|order now|subscribe|donate|"
    r"send|post|publish|submit|delete|remove|confirm|sign up|signup|register|transfer|"
    r"book now|reserve|unsubscribe|log ?out|sign out|accept)\b"
)
_PAYMENT_FIELD = re.compile(r"(?i)(cc-|card|cvc|cvv|iban|routing|account.?number|ssn)")
BROWSER_CANDIDATES = (
    ("chrome", Path("/Applications/Google Chrome.app")),
    ("msedge", Path("/Applications/Microsoft Edge.app")),
)
EXECUTABLE_CANDIDATES = (
    Path("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
)

# Collects page text and interactive elements; tags elements with data-jarvis-ref.
_SNAPSHOT_JS = r"""
(maxChars) => {
  const BLOCK_TITLE = /captcha|access denied|attention required|just a moment|are you a robot/i;
  const BLOCK_TEXT = new RegExp([
    'verify you are (a )?human', 'confirm (that )?you are (a )?human', 'made by a human',
    'are you a robot', 'unusual traffic', 'bots use', 'complete the following challenge',
    'press (and|&) hold'].join('|'), 'i');
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
  };
  const label = (el) => (el.getAttribute('aria-label') || el.innerText || el.value ||
    el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('alt') ||
    el.getAttribute('name') || '').replace(/\s+/g, ' ').trim().slice(0, 80);
  document.querySelectorAll('[data-jarvis-ref]').forEach(e => e.removeAttribute('data-jarvis-ref'));
  const nodes = document.querySelectorAll(
    'a[href], button, input:not([type=hidden]), textarea, select, ' +
    '[role=button], [role=link], [role=textbox], [role=checkbox], [contenteditable=true]');
  const elements = [];
  for (const el of nodes) {
    if (elements.length >= 60) break;
    if (!visible(el)) continue;
    const ref = 'e' + (elements.length + 1);
    el.setAttribute('data-jarvis-ref', ref);
    const tag = el.tagName.toLowerCase();
    const form = el.form || el.closest('form');
    elements.push({
      ref, tag, role: el.getAttribute('role') || '', name: label(el),
      type: (el.getAttribute('type') || '').toLowerCase(),
      href: tag === 'a' ? (el.href || '').slice(0, 200) : '',
      autocomplete: (el.getAttribute('autocomplete') || '').toLowerCase(),
      field: (el.getAttribute('name') || el.id || '').toLowerCase(),
      in_form: !!form,
      submits: (tag === 'button' && (el.type || 'submit') === 'submit' && !!form) ||
               (tag === 'input' && ['submit', 'image'].includes((el.type || '').toLowerCase())),
    });
  }
  const main = document.querySelector('main, article, [role=main]');
  let raw = main ? main.innerText : '';
  if (raw.trim().length < 200 && document.body) raw = document.body.innerText;
  const text = raw.replace(/[ \t]+/g, ' ').replace(/\n\s*\n+/g, '\n').trim();
  const captcha = !!document.querySelector(
    'iframe[src*="recaptcha"], iframe[src*="hcaptcha"], iframe[src*="turnstile"], ' +
    'iframe[src*="challenges.cloudflare"], .g-recaptcha, .h-captcha') ||
    BLOCK_TITLE.test(document.title) || BLOCK_TEXT.test(text.slice(0, 3000));
  const login = !!document.querySelector('input[type=password]');
  return {title: document.title, url: location.href, text: text.slice(0, maxChars),
          text_length: text.length, elements, captcha, login};
}
"""


def normalise_url(raw: str) -> str:
    """Add https:// to bare domains; refuse non-web schemes (file:, javascript:, ...)."""
    text = raw.strip()
    if not text or any(c in text for c in "\n\r\x00 "):
        raise ToolInputError(f"'{raw}' is not a web address.")
    scheme = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):(?!\d)", text)  # "localhost:8501" is a port
    if "://" not in text and scheme is None:
        if "." not in text.split("/")[0] and not text.startswith("localhost"):
            raise ToolInputError(
                f"'{raw}' is not a web address.", hint="Use browser.search to search."
            )
        text = ("http://" if text.startswith(("localhost", "127.0.0.1")) else "https://") + text
    parsed = urllib.parse.urlsplit(text)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise PolicyDeniedError(f"Not allowed: '{parsed.scheme}:' addresses (only http and https).")
    if not parsed.netloc:
        raise ToolInputError(f"'{raw}' has no host.")
    return text


def classify_click(element: dict[str, Any]) -> tuple[RiskLevel, str | None]:
    """HIGH for anything that submits a form or reads like a consequential action."""
    name = str(element.get("name", ""))
    if element.get("submits"):
        return RiskLevel.HIGH, "it submits a form"
    if _SENSITIVE.search(name):
        return RiskLevel.HIGH, f"'{name}' may send, buy, delete or sign you up for something"
    return RiskLevel.LOW, None


def check_typing_target(element: dict[str, Any]) -> None:
    """Refuse password and payment fields: the user must type those themselves."""
    kind = str(element.get("type", ""))
    autocomplete = str(element.get("autocomplete", ""))
    if kind == "password" or "password" in autocomplete:
        raise PolicyDeniedError(
            "Not allowed: Jarvis never types passwords.",
            hint="Type it yourself in the browser window, then ask Jarvis to continue.",
        )
    if _PAYMENT_FIELD.search(autocomplete) or _PAYMENT_FIELD.search(str(element.get("field", ""))):
        raise PolicyDeniedError("Not allowed: Jarvis never enters payment or identity numbers.")
    if element.get("tag") not in {"input", "textarea"} and element.get("role") != "textbox":
        raise ToolInputError(f"{element.get('ref')} is not a text field.")


def format_snapshot(snap: dict[str, Any], max_elements: int = 40) -> str:
    lines = [f"Page: {snap.get('title') or '(untitled)'}", f"URL: {snap.get('url')}"]
    if snap.get("captcha"):
        lines.append(
            "!! The page shows a CAPTCHA, bot check or block page: the user must handle it."
        )
    if snap.get("login"):
        lines.append("Note: the page has a password field (the user must log in themselves).")
    text = str(snap.get("text", ""))
    more = int(snap.get("text_length", len(text))) - len(text)
    lines.append("Text:\n" + text + (f"\n[… {more} more characters]" if more > 0 else ""))
    elements = snap.get("elements", [])[:max_elements]
    if elements:
        lines.append("Elements (use the ref with browser.click / browser.type):")
        for e in elements:
            kind = e["tag"] + (f"[{e['type']}]" if e.get("type") else "")
            extra = f" -> {e['href']}" if e.get("href") else ""
            lines.append(f"  {e['ref']} {kind} '{e['name']}'{extra}")
    return "\n".join(lines)


# -- browser thread -------------------------------------------------------------------------


@dataclass
class _Job:
    fn: Callable[[Any], Any]
    future: Future[Any]
    needs_page: bool = True  # False for shutdown: never launch a browser just to close it


class BrowserSession:
    """Owns Playwright on one thread; starts lazily, closes when idle."""

    def __init__(self, config: BrowserConfig, profile_dir: Path) -> None:
        self._config = config
        self._profile_dir = profile_dir
        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._page: Any = None
        self._context: Any = None
        self._playwright: Any = None
        self._idle_timer: threading.Timer | None = None

    @property
    def is_open(self) -> bool:
        return self._context is not None

    def call(self, fn: Callable[[Any], T], timeout: float = 60.0) -> T:
        """Run ``fn(page)`` on the browser thread and return its result."""
        future: Future[T] = Future()
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._loop, name="jarvis-browser", daemon=True
                )
                self._thread.start()
            if self._idle_timer is not None:
                self._idle_timer.cancel()
        self._jobs.put(_Job(fn, future))
        try:
            return future.result(timeout=timeout)
        finally:
            self._arm_idle_timer()

    def close(self, timeout: float = 10.0) -> None:
        with self._lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
            thread = self._thread
        if thread is None or not thread.is_alive():
            return
        done: Future[None] = Future()
        self._jobs.put(_Job(lambda _page: self._shutdown(), done, needs_page=False))
        self._jobs.put(None)
        thread.join(timeout=timeout)

    def _arm_idle_timer(self) -> None:
        if self._config.idle_close_seconds <= 0:
            return
        with self._lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
            self._idle_timer = threading.Timer(self._config.idle_close_seconds, self.close)
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                page = self._ensure_page() if job.needs_page else None
                job.future.set_result(job.fn(page))
            except BaseException as exc:
                job.future.set_exception(exc)

    def _ensure_page(self) -> Any:
        if self._page is not None and not self._page.is_closed():
            return self._page
        if self._context is None:
            self._launch()
        pages = self._context.pages
        self._page = pages[0] if pages else self._context.new_page()
        return self._page

    def _launch(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ToolExecutionError(
                "Playwright is not installed, so browser tools can't run.",
                hint='In the Jarvis virtualenv run: pip install -e ".[browser]"',
            ) from exc
        self._profile_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        options: dict[str, Any] = {
            "user_data_dir": str(self._profile_dir),
            "headless": self._config.headless,
            "viewport": None if not self._config.headless else {"width": 1280, "height": 900},
        }
        options.update(self._browser_choice())
        self._playwright = sync_playwright().start()
        try:
            self._context = self._playwright.chromium.launch_persistent_context(**options)
        except Exception as exc:
            self._playwright.stop()
            self._playwright = None
            raise ToolExecutionError(
                f"Could not start the browser: {str(exc).splitlines()[0][:200]}",
                hint="Install Google Chrome, set [browser] executable, or run "
                "`.venv/bin/python -m playwright install chromium` (downloads ~150 MB).",
            ) from exc
        self._context.set_default_timeout(self._config.timeout_seconds * 1000)
        log.info(
            "Browser started (%s)",
            options.get("channel") or options.get("executable_path") or "chromium",
        )

    def _browser_choice(self) -> dict[str, Any]:
        if self._config.executable:
            return {"executable_path": str(Path(self._config.executable).expanduser())}
        for channel, app in BROWSER_CANDIDATES:
            if app.exists():
                return {"channel": channel}
        for executable in EXECUTABLE_CANDIDATES:
            if executable.exists():
                return {"executable_path": str(executable)}
        return {}  # Playwright's own Chromium, if installed

    def _shutdown(self) -> None:
        # Closing a broken browser must never fail Jarvis's shutdown.
        with contextlib.suppress(Exception):
            if self._context is not None:
                self._context.close()
        with contextlib.suppress(Exception):
            if self._playwright is not None:
                self._playwright.stop()
        self._context = self._page = self._playwright = None
        log.info("Browser closed")


# -- tools -----------------------------------------------------------------------------------


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OpenArgs(_Args):
    url: str = Field(min_length=1, max_length=2000, description="web address")


class SearchArgs(_Args):
    query: str = Field(min_length=1, max_length=300)


class ReadArgs(_Args):
    max_chars: int = Field(default=3000, ge=200, le=20_000)


class ClickArgs(_Args):
    ref: str = Field(pattern=r"^e\d{1,3}$", description="element ref from the page, e.g. e3")


class TypeArgs(_Args):
    ref: str = Field(pattern=r"^e\d{1,3}$")
    text: str = Field(max_length=2000)
    submit: bool = Field(default=False, description="press Enter afterwards (submits the form)")


class EmptyArgs(_Args):
    pass


class BrowserTools:
    """Builds the ``browser.*`` tool specs around a :class:`BrowserSession`."""

    def __init__(self, session: BrowserSession, config: BrowserConfig) -> None:
        self._session = session
        self._config = config

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec("browser.open", "Open a web page and read it (text + clickable elements).",
                     OpenArgs, RiskLevel.LOW, self._prepare_open, idempotent=True),
            ToolSpec("browser.search", "Search the web and list the results.", SearchArgs,
                     RiskLevel.LOW, self._prepare_search, idempotent=True),
            ToolSpec("browser.read", "Read the current page again (text + elements).", ReadArgs,
                     RiskLevel.READ_ONLY, self._prepare_read, idempotent=True),
            ToolSpec("browser.click", "Click an element on the current page by its ref.",
                     ClickArgs, RiskLevel.LOW, self._prepare_click),
            ToolSpec("browser.type", "Type text into a field on the current page by its ref.",
                     TypeArgs, RiskLevel.LOW, self._prepare_type),
            ToolSpec("browser.close", "Close Jarvis's browser window.", EmptyArgs,
                     RiskLevel.LOW, self._prepare_close),
        ]  # fmt: skip

    # -- helpers ------------------------------------------------------------------------

    @staticmethod
    def _settle(page: Any) -> None:
        """Give scripts a moment to render (many result pages fill in after load)."""
        with contextlib.suppress(Exception):
            page.wait_for_load_state("load", timeout=5000)

    def _snapshot(self, page: Any, max_chars: int = 3000) -> dict[str, Any]:
        for _ in range(3):
            try:
                result: dict[str, Any] = page.evaluate(_SNAPSHOT_JS, max_chars)
                return result
            except Exception as exc:
                text = str(exc).splitlines()[0][:200]
                if "context was destroyed" not in text and "navigat" not in text:
                    raise ToolExecutionError(
                        f"Could not read the page: {text}", retryable=True
                    ) from exc
                self._settle(page)  # the page was navigating; read it once it has loaded
        raise ToolExecutionError("The page kept navigating; try again.", retryable=True)

    def _page_result(
        self, tool: str, snap: dict[str, Any], status_code: int | None = None
    ) -> ToolResult:
        message = format_snapshot(snap)
        if snap.get("captcha"):
            return ToolResult(
                tool=tool,
                status=ToolStatus.NEEDS_CLARIFICATION,
                message=message,
                hint="Complete the check in the browser window yourself, then ask again.",
                untrusted=True,
            )
        ok = status_code is None or status_code < 400
        return ToolResult(
            tool=tool,
            status=ToolStatus.SUCCESS if ok else ToolStatus.FAILED,
            message=message if ok else f"The page returned HTTP {status_code}.\n{message}",
            data={"url": snap.get("url"), "title": snap.get("title"), "http_status": status_code},
            untrusted=True,
            verified=ok,
            retryable=status_code is not None and status_code >= 500,
        )

    def _element(self, ref: str) -> dict[str, Any]:
        def find(page: Any) -> dict[str, Any] | None:
            snap = self._snapshot(page, 200)
            return next((e for e in snap["elements"] if e["ref"] == ref), None)

        if not self._session.is_open:
            raise ToolExecutionError("No page is open.", hint="Open a page first (browser.open).")
        element = self._session.call(find)
        if element is None:
            raise ToolExecutionError(
                f"Element {ref} isn't on the page any more.",
                hint="Read the page again (browser.read) to get fresh refs.",
            )
        return element

    def _navigate(self, page: Any, url: str) -> int | None:
        try:
            response = page.goto(url, wait_until="domcontentloaded")
        except Exception as exc:
            text = str(exc).splitlines()[0][:200]
            retryable = "Timeout" in text or "ERR_TIMED_OUT" in text
            raise ToolExecutionError(f"Could not open {url}: {text}", retryable=retryable) from exc
        return int(response.status) if response is not None else None

    # -- open / search / read -------------------------------------------------------------

    def _prepare_open(self, args: OpenArgs) -> PreparedAction:
        url = normalise_url(args.url)

        def run(_cancel: threading.Event) -> ToolResult:
            def go(page: Any) -> tuple[int | None, dict[str, Any]]:
                code = self._navigate(page, url)
                self._settle(page)
                return code, self._snapshot(page)

            code, snap = self._session.call(go)
            return self._page_result("browser.open", snap, code)

        return PreparedAction(description=f"Open {url}", run=run)

    def _prepare_search(self, args: SearchArgs) -> PreparedAction:
        url = self._config.search_url.replace("{query}", urllib.parse.quote_plus(args.query))

        def run(_cancel: threading.Event) -> ToolResult:
            def go(page: Any) -> tuple[int | None, dict[str, Any]]:
                code = self._navigate(page, url)
                self._settle(page)
                return code, self._snapshot(page)

            code, snap = self._session.call(go)
            result = self._page_result("browser.search", snap, code)
            return result.model_copy(
                update={"message": f"Search results for '{args.query}':\n{result.message}"}
            )

        return PreparedAction(description=f"Search the web for '{args.query}'", run=run)

    def _prepare_read(self, args: ReadArgs) -> PreparedAction:
        if not self._session.is_open:
            raise ToolExecutionError("No page is open.", hint="Open a page first (browser.open).")

        def run(_cancel: threading.Event) -> ToolResult:
            snap = self._session.call(lambda page: self._snapshot(page, args.max_chars))
            return self._page_result("browser.read", snap)

        return PreparedAction(description="Read the current page", run=run)

    # -- click / type -------------------------------------------------------------------

    def _prepare_click(self, args: ClickArgs) -> PreparedAction:
        element = self._element(args.ref)
        risk, why = classify_click(element)
        description = f"Click {args.ref} ({element['tag']} '{element['name']}')"

        def run(_cancel: threading.Event) -> ToolResult:
            def click(page: Any) -> dict[str, Any]:
                page.locator(f'[data-jarvis-ref="{args.ref}"]').first.click()
                with contextlib.suppress(Exception):  # not every click navigates
                    page.wait_for_load_state("domcontentloaded", timeout=10_000)
                return self._snapshot(page)

            return self._page_result("browser.click", self._session.call(click))

        return PreparedAction(
            description=description,
            run=run,
            risk=risk,
            warning=f"High impact: {why}." if why else None,
        )

    def _prepare_type(self, args: TypeArgs) -> PreparedAction:
        element = self._element(args.ref)
        check_typing_target(element)
        risk = RiskLevel.HIGH if args.submit else RiskLevel.LOW
        shown = args.text if len(args.text) <= 60 else args.text[:57] + "…"
        description = f"Type '{shown}' into {args.ref} ('{element['name']}')" + (
            " and press Enter (submits the form)" if args.submit else ""
        )

        def run(_cancel: threading.Event) -> ToolResult:
            def type_text(page: Any) -> dict[str, Any]:
                field = page.locator(f'[data-jarvis-ref="{args.ref}"]').first
                field.fill(args.text)
                if args.submit:
                    field.press("Enter")
                    with contextlib.suppress(Exception):  # not every submit navigates
                        page.wait_for_load_state("domcontentloaded", timeout=10_000)
                return self._snapshot(page)

            return self._page_result("browser.type", self._session.call(type_text))

        return PreparedAction(description=description, run=run, risk=risk)

    def _prepare_close(self, args: EmptyArgs) -> PreparedAction:
        def run(_cancel: threading.Event) -> ToolResult:
            self._session.close()
            return ToolResult(
                tool="browser.close", status=ToolStatus.SUCCESS, message="Closed the browser."
            )

        return PreparedAction(
            description="Close the browser", run=run, noop=not self._session.is_open
        )


def playwright_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("playwright") is not None
