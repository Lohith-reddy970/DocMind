"""Outbound network accounting and an opt-in local-only egress guard."""

from __future__ import annotations

import ipaddress
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx


class EgressBlockedError(RuntimeError):
    """Raised when a request would leave the local network in local-only mode."""


def is_local_host(host: str | None) -> bool:
    """Return True for loopback, private, link-local and single-label hosts."""
    if not host:
        return False
    candidate = host.strip().strip("[]").lower()
    if candidate in {"localhost", "::1"}:
        return True
    try:
        ip = ipaddress.ip_address(candidate)
    except ValueError:
        pass
    else:
        return ip.is_loopback or ip.is_private or ip.is_link_local
    if "." not in candidate:
        return True
    return candidate.endswith((".local", ".internal", ".lan", ".home"))


@dataclass
class EgressEntry:
    """A tally of outbound requests to one host for one purpose."""

    host: str
    purpose: str
    requests: int = 0
    last_at: float = field(default_factory=time.time)


class EgressLog:
    """Thread-safe record of every outbound request DocMind makes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str], EgressEntry] = {}
        self.blocked = 0

    def record(self, url: str, purpose: str) -> EgressEntry:
        host = urlparse(url).hostname or "unknown"
        key = (host, purpose)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = EgressEntry(host=host, purpose=purpose)
                self._entries[key] = entry
            entry.requests += 1
            entry.last_at = time.time()
            return entry

    def record_blocked(self) -> None:
        with self._lock:
            self.blocked += 1

    def entries(self) -> list[EgressEntry]:
        with self._lock:
            return sorted(self._entries.values(), key=lambda entry: (entry.host, entry.purpose))

    def snapshot(self) -> list[dict]:
        return [
            {
                "host": entry.host,
                "purpose": entry.purpose,
                "requests": entry.requests,
                "last_at": entry.last_at,
            }
            for entry in self.entries()
        ]

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()
            self.blocked = 0


EGRESS = EgressLog()


def make_request_hook(purpose: str, local_only: bool, log: EgressLog | None = None):
    """Build an httpx request hook that logs traffic and blocks remote hosts."""
    log = EGRESS if log is None else log

    def hook(request: httpx.Request) -> None:
        host = request.url.host
        if local_only and not is_local_host(host):
            log.record_blocked()
            raise EgressBlockedError(
                f"Blocked outbound request to '{host}': DocMind runs in local-only mode, "
                "so nothing leaves your machine. Set DOCMIND_LOCAL_ONLY=false to allow it."
            )
        log.record(str(request.url), purpose)

    return hook
