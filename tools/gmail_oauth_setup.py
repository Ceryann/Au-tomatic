"""One-time helper: obtain a Gmail OAuth refresh token (permission: send email only).

Run on YOUR computer, never in chat or in Actions. PowerShell, venv active, project folder:

    $env:GMAIL_OAUTH_CLIENT_ID = "....apps.googleusercontent.com"
    python tools/gmail_oauth_setup.py          # asks for the client secret without echoing it

A browser opens, you sign in with the Gmail account that should SEND the email and press
Allow (an "unverified app" warning is expected for a personal project: Advanced > Continue).
The refresh token is printed in this terminal only. Copy it straight into the GitHub secret
GMAIL_OAUTH_REFRESH_TOKEN. It is never written to disk and never goes to a log.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import http.server
import os
import secrets
import socketserver
import sys
import threading
import urllib.parse
import webbrowser

import requests

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/gmail.send"


def pkce_challenge(verifier: str) -> str:
    """RFC 7636 S256: base64url(sha256(verifier)) without padding."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def build_auth_url(client_id: str, redirect_uri: str, state: str, challenge: str) -> str:
    q = {
        "client_id": client_id, "redirect_uri": redirect_uri, "response_type": "code",
        "scope": SCOPE, "access_type": "offline", "prompt": "consent",
        "state": state, "code_challenge": challenge, "code_challenge_method": "S256",
    }
    return AUTH_URL + "?" + urllib.parse.urlencode(q)


class _Handler(http.server.BaseHTTPRequestHandler):
    result: dict = {}

    def do_GET(self):  # noqa: N802
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        _Handler.result = {k: v[0] for k, v in qs.items()}
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"<h3>Done. You can close this tab and return to the terminal.</h3>")

    def log_message(self, *a):  # silence request logging
        pass


def main() -> int:
    client_id = os.environ.get("GMAIL_OAUTH_CLIENT_ID", "").strip() or input("Client ID: ").strip()
    client_secret = os.environ.get("GMAIL_OAUTH_CLIENT_SECRET", "").strip() or getpass.getpass("Client secret (hidden): ").strip()
    if not client_id or not client_secret:
        print("Client ID and secret are required.")
        return 2

    verifier = secrets.token_urlsafe(64)
    state = secrets.token_urlsafe(16)
    with socketserver.TCPServer(("127.0.0.1", 0), _Handler) as httpd:
        port = httpd.server_address[1]
        redirect = f"http://127.0.0.1:{port}"
        url = build_auth_url(client_id, redirect, state, pkce_challenge(verifier))
        print("Opening the browser. If nothing opens, copy this link into it:\n" + url + "\n")
        webbrowser.open(url)
        httpd.timeout = 300
        t = threading.Thread(target=httpd.handle_request)
        t.start()
        t.join(310)

    res = _Handler.result
    if res.get("error"):
        print("Google reported an error:", res["error"])
        return 1
    if res.get("state") != state or not res.get("code"):
        print("No valid answer received (timeout or state mismatch). Run the script again.")
        return 1

    r = requests.post(TOKEN_URL, data={
        "code": res["code"], "client_id": client_id, "client_secret": client_secret,
        "redirect_uri": redirect, "grant_type": "authorization_code", "code_verifier": verifier,
    }, timeout=30)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    token = body.get("refresh_token")
    if r.status_code != 200 or not token:
        print(f"Token exchange failed (HTTP {r.status_code}, {body.get('error', 'unknown')}).")
        if r.status_code == 200:
            print("Google sent no refresh token: remove this app at myaccount.google.com/permissions and run again.")
        return 1
    print("\nSUCCESS. Copy this value into the GitHub secret GMAIL_OAUTH_REFRESH_TOKEN (and nowhere else):\n")
    print(token)
    print("\nAlso store GMAIL_OAUTH_CLIENT_ID and GMAIL_OAUTH_CLIENT_SECRET as GitHub secrets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
