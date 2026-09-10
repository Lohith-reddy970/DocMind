import httpx
import pytest

from docmind.netguard import EgressBlockedError, EgressLog, is_local_host, make_request_hook


def test_is_local_host_accepts_loopback_and_private():
    assert is_local_host("localhost")
    assert is_local_host("127.0.0.1")
    assert is_local_host("::1")
    assert is_local_host("192.168.1.10")
    assert is_local_host("10.0.0.5")
    assert is_local_host("172.17.0.2")
    assert is_local_host("[::1]")


def test_is_local_host_accepts_single_label_and_mdns():
    assert is_local_host("ollama")
    assert is_local_host("box.local")
    assert is_local_host("server.internal")


def test_is_local_host_rejects_public_hosts():
    assert not is_local_host("api.openai.com")
    assert not is_local_host("8.8.8.8")
    assert not is_local_host("example.org")
    assert not is_local_host("")
    assert not is_local_host(None)


def test_egress_log_records_by_host_and_purpose():
    log = EgressLog()
    hook = make_request_hook("generation", local_only=True, log=log)
    hook(httpx.Request("POST", "http://localhost:11434/api/chat"))
    hook(httpx.Request("POST", "http://localhost:11434/api/chat"))
    entries = log.entries()
    assert len(entries) == 1
    assert entries[0].host == "localhost"
    assert entries[0].purpose == "generation"
    assert entries[0].requests == 2
    assert log.blocked == 0


def test_hook_blocks_remote_hosts_in_local_only_mode():
    log = EgressLog()
    hook = make_request_hook("generation", local_only=True, log=log)
    with pytest.raises(EgressBlockedError):
        hook(httpx.Request("POST", "https://api.openai.com/v1/chat"))
    assert log.blocked == 1
    assert log.entries() == []


def test_hook_allows_remote_when_guard_disabled():
    log = EgressLog()
    hook = make_request_hook("generation", local_only=False, log=log)
    hook(httpx.Request("POST", "https://example.com/api"))
    assert log.entries()[0].host == "example.com"


def test_snapshot_and_reset():
    log = EgressLog()
    log.record("http://localhost:11434/api/embed", "embeddings")
    log.record_blocked()
    assert log.snapshot()[0]["purpose"] == "embeddings"
    assert log.blocked == 1
    log.reset()
    assert log.entries() == []
    assert log.blocked == 0
