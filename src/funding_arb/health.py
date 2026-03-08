"""Lightweight async HTTP health check endpoint."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from .executor import KILL_SWITCH_PATH

logger = logging.getLogger(__name__)


class HealthCheck:
    """Minimal HTTP health check server for monitoring.

    Responds to GET / with JSON containing:
    - status: "ok" or "degraded"
    - uptime_s: seconds since start
    - last_tick_s: seconds since last successful tick
    - kill_switch: whether kill switch is active
    - open_positions: count of open positions
    """

    def __init__(self, port: int = 8080):
        self.port = port
        self._start_time = datetime.now(UTC)
        self._last_tick: datetime | None = None
        self._open_positions = 0
        self._last_tick_ok = True
        self._server: asyncio.AbstractServer | None = None

    def record_tick(self, success: bool = True, open_positions: int = 0) -> None:
        """Record that a tick completed."""
        self._last_tick = datetime.now(UTC)
        self._last_tick_ok = success
        self._open_positions = open_positions

    def _build_response(self) -> dict[str, Any]:
        now = datetime.now(UTC)
        uptime = (now - self._start_time).total_seconds()

        last_tick_s = None
        if self._last_tick:
            last_tick_s = (now - self._last_tick).total_seconds()

        kill_active = KILL_SWITCH_PATH.exists()

        # Degraded if: no tick in 5 minutes, kill switch on, or last tick failed
        is_degraded = (
            kill_active
            or not self._last_tick_ok
            or (last_tick_s is not None and last_tick_s > 300)
        )

        return {
            "status": "degraded" if is_degraded else "ok",
            "uptime_s": round(uptime, 1),
            "last_tick_s": round(last_tick_s, 1) if last_tick_s is not None else None,
            "kill_switch": kill_active,
            "open_positions": self._open_positions,
            "timestamp": now.isoformat(),
        }

    async def _handle_request(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            # Read the request line (we only care that something came in)
            await asyncio.wait_for(reader.readline(), timeout=5.0)
            # Drain remaining headers
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=2.0)
                if line in (b"\r\n", b"\n", b""):
                    break

            body = json.dumps(self._build_response())
            response = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Connection: close\r\n"
                f"\r\n"
                f"{body}"
            )
            writer.write(response.encode())
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    async def start(self) -> None:
        """Start the health check HTTP server."""
        try:
            self._server = await asyncio.start_server(
                self._handle_request, "0.0.0.0", self.port
            )
            logger.info(f"Health check server started on :{self.port}")
        except OSError as e:
            logger.warning(f"Could not start health check on :{self.port}: {e}")

    async def stop(self) -> None:
        """Stop the health check server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
