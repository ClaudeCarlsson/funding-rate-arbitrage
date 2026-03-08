"""Tests for the health check endpoint."""
from __future__ import annotations

import asyncio
import json

import pytest

from funding_arb.executor import KILL_SWITCH_PATH
from funding_arb.health import HealthCheck


class TestHealthCheckResponse:
    def test_initial_status_ok(self):
        hc = HealthCheck()
        resp = hc._build_response()
        assert resp["status"] == "ok"
        assert resp["kill_switch"] is False
        assert resp["open_positions"] == 0
        assert resp["last_tick_s"] is None
        assert "uptime_s" in resp
        assert "timestamp" in resp

    def test_status_after_tick(self):
        hc = HealthCheck()
        hc.record_tick(success=True, open_positions=2)
        resp = hc._build_response()
        assert resp["status"] == "ok"
        assert resp["open_positions"] == 2
        assert resp["last_tick_s"] is not None
        assert resp["last_tick_s"] < 5  # should be near-instant

    def test_status_degraded_after_failed_tick(self):
        hc = HealthCheck()
        hc.record_tick(success=False)
        resp = hc._build_response()
        assert resp["status"] == "degraded"

    def test_status_degraded_with_kill_switch(self):
        hc = HealthCheck()
        hc.record_tick(success=True)
        try:
            KILL_SWITCH_PATH.touch()
            resp = hc._build_response()
            assert resp["status"] == "degraded"
            assert resp["kill_switch"] is True
        finally:
            KILL_SWITCH_PATH.unlink(missing_ok=True)

    def test_open_positions_tracked(self):
        hc = HealthCheck()
        hc.record_tick(success=True, open_positions=0)
        assert hc._build_response()["open_positions"] == 0
        hc.record_tick(success=True, open_positions=3)
        assert hc._build_response()["open_positions"] == 3


class TestHealthCheckServer:
    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        hc = HealthCheck(port=18080)
        await hc.start()
        assert hc._server is not None
        await hc.stop()
        assert hc._server is None

    @pytest.mark.asyncio
    async def test_http_response(self):
        hc = HealthCheck(port=18081)
        await hc.start()
        try:
            hc.record_tick(success=True, open_positions=1)

            reader, writer = await asyncio.open_connection("127.0.0.1", 18081)
            writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()

            data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            writer.close()

            # Parse the HTTP response
            text = data.decode()
            body = text.split("\r\n\r\n", 1)[1]
            parsed = json.loads(body)

            assert parsed["status"] == "ok"
            assert parsed["open_positions"] == 1
        finally:
            await hc.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self):
        hc = HealthCheck()
        await hc.stop()  # should not raise
