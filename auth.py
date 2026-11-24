"""
Access token gate for the dashboard, live feeds and camera ingest.

Exposing the server through a tunnel (ngrok, cloudflared) puts it on the public
internet: without a gate, anyone holding the URL could watch every camera or
push fake frames into a room. So:

  * Direct requests from this machine (loopback, with no proxy headers) are
    always allowed, so http://localhost:8000 keeps working with zero setup.
  * Everything else (the tunnel, other LAN devices) must present the token,
    either as ?token=... or as the `panic_token` cookie. A valid ?token= sets
    the cookie, so a phone that opened the QR link once stays signed in, and
    the page's own fetch()/WebSocket calls are covered automatically.

Tunnels connect to uvicorn from 127.0.0.1, so "is the peer loopback?" alone is
not enough. Every tunnel adds forwarding headers (X-Forwarded-For, and
Cf-Connecting-Ip for cloudflared), and uvicorn's proxy-header handling also
rewrites the client address from them. A loopback peer WITHOUT any of those
headers is a local browser, so either signal on its own is enough to require
the token.
"""
import hmac
import os
import secrets
from http.cookies import SimpleCookie
from urllib.parse import parse_qs

COOKIE = "panic_token"
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
PROXY_HEADERS = {b"x-forwarded-for", b"x-forwarded-host", b"forwarded",
                 b"cf-connecting-ip", b"x-real-ip"}


def load_token() -> tuple[str, bool]:
    """(token, generated). PANIC_TOKEN if set, otherwise a random one per run."""
    tok = os.environ.get("PANIC_TOKEN", "").strip()
    if tok:
        return tok, False
    return secrets.token_urlsafe(9), True


def _headers(scope):
    return {k.lower(): v for k, v in scope.get("headers", [])}


def is_direct_local(scope) -> bool:
    client = scope.get("client") or ("", 0)
    if client[0] not in LOOPBACK:
        return False
    return not (PROXY_HEADERS & set(_headers(scope)))


def _query_token(scope):
    qs = parse_qs(scope.get("query_string", b"").decode("latin-1"))
    vals = qs.get("token")
    return vals[0] if vals else None


def _cookie_token(scope):
    raw = _headers(scope).get(b"cookie")
    if not raw:
        return None
    jar = SimpleCookie()
    try:
        jar.load(raw.decode("latin-1"))
    except Exception:
        return None
    m = jar.get(COOKIE)
    return m.value if m else None


class TokenGate:
    """Pure ASGI middleware, so it covers WebSockets as well as HTTP
    (Starlette's BaseHTTPMiddleware only sees HTTP)."""

    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    def _ok(self, candidate) -> bool:
        return bool(candidate) and hmac.compare_digest(candidate.encode(), self.token.encode())

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        q = _query_token(scope)
        allowed = is_direct_local(scope) or self._ok(q) or self._ok(_cookie_token(scope))
        if allowed:
            if scope["type"] == "http" and self._ok(q):
                send = self._with_cookie(send, scope)
            return await self.app(scope, receive, send)

        if scope["type"] == "websocket":
            # Accept, then close with an application code: a handshake rejected
            # outright shows up in the browser only as an anonymous 1006, and the
            # phone page needs to tell "wrong token" apart from "network blip".
            msg = await receive()
            if msg["type"] == "websocket.connect":
                await send({"type": "websocket.accept"})
                await send({"type": "websocket.close", "code": 4401,
                            "reason": "access token required"})
            return

        body = (b"<!doctype html><meta name=viewport content='width=device-width'>"
                b"<body style='font-family:sans-serif;background:#0e1117;color:#e6edf3;"
                b"padding:24px'><h2>Access token required</h2><p>Open this page with the "
                b"link or QR code shown on the dashboard (it includes <code>?token=</code>). "
                b"The token is printed in the server console at startup.</p>")
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"text/html; charset=utf-8"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    def _with_cookie(self, send, scope):
        secure = _headers(scope).get(b"x-forwarded-proto", b"") == b"https" \
            or scope.get("scheme") == "https"
        cookie = f"{COOKIE}={self.token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000"
        if secure:
            cookie += "; Secure"

        async def wrapped(message):
            if message["type"] == "http.response.start":
                message = dict(message)
                message["headers"] = list(message.get("headers", [])) + [
                    (b"set-cookie", cookie.encode())]
            await send(message)
        return wrapped
