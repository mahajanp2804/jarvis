"""Messages and email: text someone, write an email, or open a draft.

* **Sending always asks first** (HIGH), every time: the confirmation shows the
  person, the exact number or address, and the exact text. A "yes" is never
  remembered for next time.
* The recipient is worked out before asking (see
  :func:`jarvis.tools.contacts.resolve_recipient`): when several people or
  numbers match, Jarvis asks which one instead of guessing.
* ``mail.draft`` only opens a filled-in email for you to read and send
  yourself, so it's LOW risk.
* Messages can't confirm delivery, so the reply says the message was sent,
  not that it arrived.
"""

from __future__ import annotations

import threading

from pydantic import BaseModel, ConfigDict, Field

from jarvis.api.schemas import RiskLevel, ToolResult, ToolStatus
from jarvis.platforms.base import PlatformAdapter
from jarvis.tools.base import PreparedAction, ToolSpec
from jarvis.tools.contacts import resolve_recipient

PREVIEW = 600


class MessageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    to: str = Field(min_length=1, max_length=80, description='a name, "mom", or a number')
    text: str = Field(min_length=1, max_length=2000)


class EmailArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    to: str = Field(min_length=1, max_length=80, description='a name, "my manager", or an address')
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(default="", max_length=20_000)


def _quote(text: str) -> str:
    shown = text if len(text) <= PREVIEW else text[:PREVIEW] + "…"
    return f"“{shown}”"


class MessagingTools:
    """Builds the ``messages.*`` and ``mail.*`` tool specs."""

    def __init__(self, platform: PlatformAdapter) -> None:
        self._platform = platform

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec("messages.send", "Send a text message with Messages (always asks first).",
                     MessageArgs, RiskLevel.HIGH, self._prepare_message),
            ToolSpec("mail.draft", "Open an email in Mail, filled in, for the user to send.",
                     EmailArgs, RiskLevel.LOW, self._prepare_draft),
            ToolSpec("mail.send", "Send an email with Mail (always asks first).", EmailArgs,
                     RiskLevel.HIGH, self._prepare_email),
        ]  # fmt: skip

    def _prepare_message(self, args: MessageArgs) -> PreparedAction:
        recipient = resolve_recipient(self._platform.contacts(), args.to, "text")
        messenger = self._platform.messages()
        text = args.text.strip()

        def run(cancel: threading.Event) -> ToolResult:
            service = messenger.send(recipient.handle, text)
            return ToolResult(
                tool="messages.send",
                status=ToolStatus.SUCCESS,
                message=f"Sent your message to {recipient.name} by {service}.",
                data={"to": recipient.handle, "service": service},
            )

        return PreparedAction(
            description=f"Send a message to {recipient.describe()}: {_quote(text)}",
            run=run,
            warning="It goes out as soon as you say yes and can't be unsent.",
        )

    def _prepare_mail(self, args: EmailArgs, *, send: bool) -> PreparedAction:
        recipient = resolve_recipient(self._platform.contacts(), args.to, "email")
        mail = self._platform.mail()
        subject, body = " ".join(args.subject.split()), args.body.strip()
        tool = "mail.send" if send else "mail.draft"

        def run(cancel: threading.Event) -> ToolResult:
            mail.compose(recipient.handle, subject, body, send=send)
            message = (
                f"Sent the email “{subject}” to {recipient.name}."
                if send
                else f"Opened an email to {recipient.name} in Mail: check it and press Send."
            )
            return ToolResult(
                tool=tool,
                status=ToolStatus.SUCCESS,
                message=message,
                data={"to": recipient.handle, "sent": send},
            )

        verb = "Send an email" if send else "Open an email"
        content = f", saying {_quote(body)}" if body else ""
        return PreparedAction(
            description=f"{verb} to {recipient.describe()}: “{subject}”{content}",
            run=run,
            warning="It goes out as soon as you say yes and can't be unsent." if send else None,
        )

    def _prepare_draft(self, args: EmailArgs) -> PreparedAction:
        return self._prepare_mail(args, send=False)

    def _prepare_email(self, args: EmailArgs) -> PreparedAction:
        return self._prepare_mail(args, send=True)
