"""Diagnose WHY Gmail drops the SMTP login. Stdlib only. Prints stages and error TYPES only:
never the address, never the password (this runs in a PUBLIC repo's Actions log too).

Local (PowerShell, venv active):
    $env:GMAIL_USER = "..."; $env:GMAIL_APP_PASSWORD = "..."     # in the terminal, never in chat
    python tools/probe_smtp.py

Reading the result: compare line "REAL account" with line "BOGUS account" for the same port.
  BOGUS -> 535 (server answers)  and REAL -> disconnect : the block is specific to this account/secret.
  BOGUS -> disconnect too                                : Google drops logins from this network (IP block).
  REAL  -> OK                                            : credentials fine; earlier failure was transient.
  REAL  -> 535                                           : wrong user or app password.
"""

from __future__ import annotations

import os
import smtplib
import ssl
import string
import sys

HOST = "smtp.gmail.com"
BOGUS_USER = "au.tomatic.probe.nonexistent.7391@gmail.com"
BOGUS_PASS = "abcdefghijklmnop"


def classify(exc: BaseException) -> str:
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        return f"server answered: login rejected (code {exc.smtp_code})"
    if isinstance(exc, smtplib.SMTPServerDisconnected):
        return "server CLOSED the connection (no answer)"
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return f"network error ({type(exc).__name__})"
    return f"{type(exc).__name__}"


def try_login(mode: str, user: str, password: str, *, factory_ssl=smtplib.SMTP_SSL, factory=smtplib.SMTP) -> str:
    ctx = ssl.create_default_context()
    stage = "connect"
    try:
        if mode == "ssl465":
            srv = factory_ssl(HOST, 465, context=ctx, timeout=30)
        else:
            srv = factory(HOST, 587, timeout=30)
        with srv:
            if mode == "starttls587":
                stage = "ehlo"
                srv.ehlo()
                stage = "starttls"
                srv.starttls(context=ctx)
            stage = "ehlo"
            srv.ehlo()
            stage = "login"
            srv.login(user, password)
        return "OK: login accepted"
    except Exception as exc:  # noqa: BLE001 - diagnostic tool, report anything
        return f"FAILED at stage '{stage}': {classify(exc)}"


def describe_secret(password: str) -> str:
    cleaned = "".join(password.split())
    ok_chars = all(c in string.ascii_lowercase for c in cleaned)
    return (f"length after removing whitespace = {len(cleaned)} (Google app password: 16), "
            f"had stray whitespace = {cleaned != password}, only a-z = {ok_chars}")


def main() -> int:
    user = os.environ.get("GMAIL_USER", "").strip()
    raw = os.environ.get("GMAIL_APP_PASSWORD", "")
    print("GMAIL_USER set:", bool(user), "| contains one '@':", user.count("@") == 1)
    print("GMAIL_APP_PASSWORD set:", bool(raw), "|", describe_secret(raw))
    password = "".join(raw.split())
    for mode in ("ssl465", "starttls587"):
        print(f"[{mode}] BOGUS account: {try_login(mode, BOGUS_USER, BOGUS_PASS)}")
        if user and password:
            print(f"[{mode}] REAL  account: {try_login(mode, user, password)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
