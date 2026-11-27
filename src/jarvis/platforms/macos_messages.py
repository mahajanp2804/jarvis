"""Messages and Mail on macOS, through fixed AppleScripts.

* The recipient and the text go in ``argv``; nothing in them can change what
  the script does.
* Messages tries iMessage first, then SMS (which needs Text Message
  Forwarding from an iPhone). AppleScript can't tell whether a message was
  delivered, only that Messages accepted it.
* Mail either sends the message or opens it in a compose window, filled in,
  for you to read and send yourself.
"""

from __future__ import annotations

import subprocess

from jarvis.errors import ToolExecutionError
from jarvis.platforms.applescript import AppleScript, Runner
from jarvis.platforms.base import MailApp, Messenger

SEND_MESSAGE = """\
on run argv
	set {theHandle, theText, serviceName} to items 1 thru 3 of argv
	tell application "Messages"
		if serviceName is "SMS" then
			set svc to 1st account whose service type = SMS
		else
			set svc to 1st account whose service type = iMessage
		end if
		send theText to participant theHandle of svc
	end tell
end run"""

COMPOSE_MAIL = """\
on run argv
	set {theTo, theSubject, theBody, sendNow} to items 1 thru 4 of argv
	tell application "Mail"
		set props to {subject:theSubject, content:theBody, visible:(sendNow is not "1")}
		set m to make new outgoing message with properties props
		tell m to make new to recipient at end of to recipients with properties {address:theTo}
		if sendNow is "1" then
			send m
		else
			activate
		end if
	end tell
end run"""

ALL_SCRIPTS = (SEND_MESSAGE, COMPOSE_MAIL)


class MacMessages(Messenger):
    def __init__(self, runner: Runner = subprocess.run) -> None:
        self._script = AppleScript("Messages", runner)

    def send(self, handle: str, text: str) -> str:
        try:
            self._script(SEND_MESSAGE, f"send the message to {handle}", handle, text, "iMessage")
            return "iMessage"
        except ToolExecutionError as first:  # not on iMessage: try SMS through the iPhone
            try:
                self._script(SEND_MESSAGE, f"send the message to {handle}", handle, text, "SMS")
            except ToolExecutionError:
                raise first from None
            return "SMS"


class MacMail(MailApp):
    def __init__(self, runner: Runner = subprocess.run) -> None:
        self._script = AppleScript("Mail", runner)

    def compose(self, to: str, subject: str, body: str, *, send: bool) -> None:
        what = f"send the email to {to}" if send else f"open an email to {to}"
        self._script(COMPOSE_MAIL, what, to, subject, body, "1" if send else "0")
