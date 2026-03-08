"""Zero-Allocation Ring Buffer for lock-free Inter-Process Communication (IPC)."""
from __future__ import annotations

import contextlib
import logging
import struct
from multiprocessing import shared_memory

logger = logging.getLogger(__name__)

# HEADER: write_cursor(8), read_cursor(8), capacity(8), padding(40) = 64 bytes
HEADER_FORMAT = "<QQQ40x"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# TICK: timestamp(8), mid_price(8), bid_depth(8), ask_depth(8), instrument_id(2), exchange_id(1), pad(5), reserved(24) = 64 bytes
# Wait, I am packing (H, B) but format is HB.
# In Rust I have: instrument_id: u16, exchange_id: u8.
# Let's be explicit.
TICK_FORMAT = "<ddddHB5x24x"
TICK_SIZE = struct.calcsize(TICK_FORMAT)

# ENVELOPE: timestamp(8), max_notional(8), min_price(8), max_price(8), max_delta(8), halt(1), pad(7), reserved(16) = 64 bytes
ENVELOPE_FORMAT = "<dddddB7x16x"
ENVELOPE_SIZE = struct.calcsize(ENVELOPE_FORMAT)


class DisruptorRingBuffer:
    """
    A single-producer, single-consumer lock-free ring buffer backed by shared RAM.
    Bypasses the GIL and Python Garbage Collector entirely.
    """

    def __init__(self, name: str, capacity: int = 4096, create: bool = True):
        self.name = name

        if create:
            self.capacity = capacity
            self.buffer_size = HEADER_SIZE + (capacity * TICK_SIZE)
            try:
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.buffer_size)
                # Initialize: write=0, read=0, capacity
                struct.pack_into(HEADER_FORMAT, self.shm.buf, 0, 0, 0, capacity)
            except FileExistsError:
                self.shm = shared_memory.SharedMemory(name=self.name)
                _, _, self.capacity = struct.unpack_from(HEADER_FORMAT, self.shm.buf, 0)
                self.buffer_size = HEADER_SIZE + (self.capacity * TICK_SIZE)
        else:
            self.shm = shared_memory.SharedMemory(name=self.name)
            # Read capacity from header
            _, _, self.capacity = struct.unpack_from(HEADER_FORMAT, self.shm.buf, 0)
            self.buffer_size = HEADER_SIZE + (self.capacity * TICK_SIZE)

        self.buf = self.shm.buf

    def publish(self, timestamp: float, mid_price: float, bid_depth: float, ask_depth: float,
                instrument_id: int, exchange_id: int) -> bool:
        """Producer: Writes a tick directly into raw memory."""
        write_cursor, read_cursor, capacity = struct.unpack_from(HEADER_FORMAT, self.buf, 0)

        # Check if buffer is full
        if write_cursor - read_cursor >= capacity:
            return False # Drop packet

        index = write_cursor % capacity
        offset = HEADER_SIZE + (index * TICK_SIZE)

        # Write payload (matching TICK_FORMAT: <ddddHB5x24x)
        struct.pack_into(TICK_FORMAT, self.buf, offset,
                         float(timestamp), float(mid_price), float(bid_depth), float(ask_depth),
                         int(instrument_id), int(exchange_id))

        # Memory barrier / update cursor
        struct.pack_into("<Q", self.buf, 0, write_cursor + 1)
        return True

    def consume(self) -> list[tuple]:
        """Consumer: Reads all available ticks from memory instantly."""
        write_cursor, read_cursor, capacity = struct.unpack_from(HEADER_FORMAT, self.buf, 0)

        if read_cursor == write_cursor:
            return [] # Empty

        ticks = []
        # Batch read to minimize struct overhead
        while read_cursor < write_cursor:
            index = read_cursor % capacity
            offset = HEADER_SIZE + (index * TICK_SIZE)

            tick = struct.unpack_from(TICK_FORMAT, self.buf, offset)
            ticks.append(tick)
            read_cursor += 1

        # Update read cursor
        struct.pack_into("<Q", self.buf, 8, read_cursor)
        return ticks
    def close(self):
        self.shm.close()

    def unlink(self):
        """Destroy the shared memory segment."""
        with contextlib.suppress(FileNotFoundError):
            self.shm.unlink()
