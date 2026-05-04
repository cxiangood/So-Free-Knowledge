from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import requests

import llm.client as client_mod


def _reset_rate_limit_state() -> None:
    with client_mod.RATE_LIMIT_LOCK:
        client_mod.RATE_LIMIT_ACTIVE = False
        client_mod.RATE_LIMIT_UNTIL = 0.0


def _http_429_error() -> requests.HTTPError:
    response = requests.Response()
    response.status_code = 429
    return requests.HTTPError("429 Too Many Requests", response=response)


def test_cooldown_not_extended_when_already_active(monkeypatch) -> None:
    _reset_rate_limit_state()
    fake_now = {"value": 100.0}
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: fake_now["value"])
    monkeypatch.setattr(client_mod, "RATE_LIMIT_COOLDOWN_SECONDS", 60.0)
    monkeypatch.setattr(client_mod, "RATE_LIMIT_SAFETY_SECONDS", 0.0)

    client_mod._enter_rate_limit_cooldown()
    first_until = client_mod.RATE_LIMIT_UNTIL
    fake_now["value"] = 110.0
    client_mod._enter_rate_limit_cooldown()
    second_until = client_mod.RATE_LIMIT_UNTIL

    assert first_until == 160.0
    assert second_until == first_until


def test_wait_if_rate_limited_blocks_remaining_time(monkeypatch) -> None:
    _reset_rate_limit_state()
    lock = Lock()
    fake_now = {"value": 0.0}
    sleep_calls: list[float] = []

    def fake_monotonic() -> float:
        with lock:
            return fake_now["value"]

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        with lock:
            fake_now["value"] += seconds

    monkeypatch.setattr(client_mod.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(client_mod.time, "sleep", fake_sleep)
    monkeypatch.setattr(client_mod, "RATE_LIMIT_COOLDOWN_SECONDS", 60.0)
    monkeypatch.setattr(client_mod, "RATE_LIMIT_SAFETY_SECONDS", 0.0)

    # t=0 opens cooldown to t=60
    client_mod._enter_rate_limit_cooldown()

    # t=10 caller should wait 50
    with lock:
        fake_now["value"] = 10.0
    client_mod._wait_if_rate_limited()
    assert sleep_calls[-1] == 50.0
    assert client_mod.RATE_LIMIT_ACTIVE is False

    # reopen and test t=20 caller waits 40
    _reset_rate_limit_state()
    with lock:
        fake_now["value"] = 0.0
    client_mod._enter_rate_limit_cooldown()
    with lock:
        fake_now["value"] = 20.0
    client_mod._wait_if_rate_limited()
    assert sleep_calls[-1] == 40.0
    assert client_mod.RATE_LIMIT_ACTIVE is False


def test_build_reply_rate_limit_retries_with_shared_window(monkeypatch) -> None:
    _reset_rate_limit_state()
    monkeypatch.setattr(client_mod, "RATE_LIMIT_COOLDOWN_SECONDS", 60.0)
    monkeypatch.setattr(client_mod, "RATE_LIMIT_SAFETY_SECONDS", 0.0)

    lock = Lock()
    fake_now = {"value": 0.0}
    first_429_done = Event()
    post_calls = {"count": 0}
    sleep_calls: list[float] = []

    def fake_monotonic() -> float:
        with lock:
            return fake_now["value"]

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        with lock:
            fake_now["value"] += seconds

    real_wait = client_mod._wait_if_rate_limited

    def wrapped_wait() -> None:
        real_wait()

    def fake_post(*args, **kwargs):
        del args, kwargs
        with lock:
            post_calls["count"] += 1
            idx = post_calls["count"]
            current = fake_now["value"]
        if idx == 1:
            # first thread at t=0 gets 429
            first_429_done.set()
            raise _http_429_error()
        if idx == 2:
            # second thread request starts after cooldown, success
            assert current >= 60.0
            class _Resp:
                def raise_for_status(self):
                    return None

                @staticmethod
                def json():
                    return {"choices": [{"message": {"content": "ok"}}]}

            return _Resp()
        class _Resp:
            def raise_for_status(self):
                return None

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "ok"}}]}

        return _Resp()

    monkeypatch.setattr(client_mod.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(client_mod.time, "sleep", fake_sleep)
    monkeypatch.setattr(client_mod, "_wait_if_rate_limited", wrapped_wait)
    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    cfg = client_mod.LLMConfig(api_key="k", model_id="m", base_url="https://x")
    client = client_mod.LLMClient(cfg)

    def worker1() -> str:
        return client.build_reply("s", "u1")

    def worker2() -> str:
        first_429_done.wait(timeout=2)
        with lock:
            fake_now["value"] = 10.0
        return client.build_reply("s", "u2")

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(worker1)
        f2 = pool.submit(worker2)
        r1 = f1.result(timeout=5)
        r2 = f2.result(timeout=5)

    assert r1 == "ok"
    assert r2 == "ok"
    assert any(abs(v - 60.0) < 0.01 for v in sleep_calls)
