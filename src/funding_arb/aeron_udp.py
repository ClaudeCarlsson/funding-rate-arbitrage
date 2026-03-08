"""UDP Multicast Networking Layer (inspired by LMAX Aeron) for Geographic IPC."""
from __future__ import annotations

import json
import logging
import socket
import struct
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class UDPConfig:
    multicast_group: str = '224.0.0.1'
    port: int = 10000
    ttl: int = 2  # Time-To-Live for multicast (determines geographic reach)

class AeronUDPNode:
    """
    High-frequency UDP Multicast node.
    Bypasses TCP handshakes for microsecond latency between the Brain Node and Execution Nodes.
    """

    def __init__(self, config: UDPConfig | None = None, is_publisher: bool = False):
        self.config = config or UDPConfig()
        self.is_publisher = is_publisher

        # Create the raw UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

        # Optimize socket buffers for high throughput
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        if self.is_publisher:
            # Set the Time-to-Live for multicast packets
            ttl_bin = struct.pack('@i', self.config.ttl)
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl_bin)
        else:
            # Bind to the port
            self.sock.bind(('', self.config.port))
            # Tell the OS to add the socket to the multicast group
            mreq = struct.pack("4sl", socket.inet_aton(self.config.multicast_group), socket.INADDR_ANY)
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            # Non-blocking for high-frequency polling
            self.sock.setblocking(False)

    def publish_raft_log(self, term: int, index: int, command: dict) -> None:
        """Serialize and multicast a Raft consensus log entry."""
        if not self.is_publisher:
            raise RuntimeError("Cannot publish from a listener node.")

        payload = {
            "type": "RAFT_APPEND",
            "term": term,
            "index": index,
            "command": command
        }

        # For maximum speed in production, this would be a custom binary struct.
        # JSON is used here for structural clarity.
        raw_bytes = json.dumps(payload).encode('utf-8')

        self.sock.sendto(raw_bytes, (self.config.multicast_group, self.config.port))

    def poll(self) -> dict | None:
        """Non-blocking read of the UDP buffer."""
        try:
            data, _address = self.sock.recvfrom(4096)
            return json.loads(data.decode('utf-8'))
        except BlockingIOError:
            # No data currently on the NIC buffer
            return None
        except Exception as e:
            logger.error(f"UDP Read Error: {e}")
            return None

    def close(self):
        self.sock.close()
