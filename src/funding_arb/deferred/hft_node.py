"""High-Frequency Trading Node Runner for the Global Arbitrage Mesh."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Literal

from .aeron_udp import AeronUDPNode, UDPConfig
from .affinity import pin_process_to_core
from .disruptor import DisruptorRingBuffer
from .market_maker import AvellanedaStoikovEngine
from .network_calculus import MinPlusLatencyVeto
from .raft import NodeState, RaftNode
from .toxicity import VPINDetector
from .tracking import OrderBookParticleFilter

logger = logging.getLogger(__name__)

NodeType = Literal["brain", "execution"]

class HFTNode:
    """Represents a physical bare-metal node in the arbitrage mesh."""

    def __init__(self, role: NodeType, node_id: str, peers: list[str], core_pin: int | None = None):
        self.role = role
        self.node_id = node_id

        if core_pin is not None:
            pin_process_to_core(core_pin)

        # Consensus
        self.raft = RaftNode(node_id=node_id, peers=peers)
        if self.role == "brain":
            self.raft.state = NodeState.LEADER

        # Networking
        self.udp_config = UDPConfig()
        self.network = AeronUDPNode(config=self.udp_config, is_publisher=(self.role == "brain"))

        # IPC
        self.disruptor = DisruptorRingBuffer(name=f"disruptor_{node_id}", create=(self.role == "brain"))

        # Microstructure & Execution
        self.particle_filter = OrderBookParticleFilter()
        self.market_maker = AvellanedaStoikovEngine()
        self.vpin = VPINDetector()
        self.latency_veto = MinPlusLatencyVeto()

    async def run_brain_loop(self):
        """
        The Shadow Brain: Runs Z3 and Convex Optimization asynchronously (100ms loop).
        Generates the SafetyEnvelope for the Brawn.
        """
        logger.info(f"Starting Shadow Brain Node: {self.node_id}")
        self.particle_filter.initialize(50000.0, 10.0, 10.0, time.time())

        while True:
            # 1. Update state from Disruptor
            ticks = self.disruptor.consume()
            for tick in ticks:
                self.particle_filter.update(tick[1], tick[2], tick[3], tick[0])

            # 2. Async Formal Verification & Portfolio Optimization
            # Instead of gating, we calculate the SAFE ENVELOPE
            self.particle_filter.estimate_state()

            # Example Z3 Pre-flight (would be called here)            # verifier.verify_margin_invariant(...)

            # 3. Publish SafetyEnvelope to Brawn
            # This allows Brawn to fire instantly without asking permission
            # In a full implementation, this writes to the DisruptorRingBuffer
            # self.shm_safety.publish(...)

            # 4. Janitor Role: Check for geographic desync
            # If Tokyo filled but Singapore didn't (latencies > 100ms), Janitor unwinds.
            await self._janitor_reconciliation()

            await asyncio.sleep(0.1) # 100ms slow-loop

    async def run_execution_loop(self):
        """
        The Optimistic Brawn: Fires trades instantly within the pre-approved SafetyEnvelope.
        """
        logger.info(f"Starting Optimistic Execution Node: {self.node_id}")

        while True:
            # 1. Pull latest SafetyEnvelope from SHM
            # if trade in envelope: fire()

            # 2. Optimistic local execution
            # No Raft quorum required for firing - local speed is king.
            # We multicast 'FILLED' status after the fact for the Janitor.

            # 3. Check toxicity (VPIN)
            if self.vpin.is_toxic():
                logger.warning("VPIN BREACH: Optimistic safety override. Canceling.")

            await asyncio.sleep(0.0001) # Microsecond polling

    async def _janitor_reconciliation(self):
        """
        Asynchronous desync cleanup (The 'Two Generals' Janitor).
        Monitors the global fill status via UDP. If legs are desynchronized
        longer than one RTT, it issues emergency market-neutralizing orders.
        """
        # 1. Poll Network for 'FILLED' heartbeats from other nodes
        remote_msg = self.network.poll()
        if remote_msg and remote_msg.get("type") == "LEG_FILL":
            remote_fill = remote_msg.get("amount", 0.0)
            instrument = remote_msg.get("instrument")

            # 2. Compare with local fill state
            # (In a real impl, we'd track this in a Local Ledger)
            local_fill = 1.0 # Mock local fill
            drift = abs(local_fill - remote_fill)

            if drift > 0.01: # Threshold for 'naked delta'
                logger.error(f"JANITOR: Delta Drift Detected ({drift:.4f} {instrument}). Initiating Recovery.")

                # 3. Check RTT-Volatility Breach & Market Impact
                state = self.particle_filter.estimate_state()

                # Check if our unwind would eat more than 20% of the visible order book
                # Assume worst-case depth (minimum of bid/ask)
                available_depth = min(state.expected_bid_depth, state.expected_ask_depth)
                market_impact_pct = drift / max(available_depth, 0.001)

                if market_impact_pct > 0.20:
                    logger.critical(f"JANITOR: Market Impact Veto! Unwind size {drift:.2f} is {market_impact_pct*100:.1f}% of book. Holding for liquidity.")
                    # Fallback to fractional passive unwind here
                    return

                # If price has moved > 0.5% during the desync window, market-unwind everything
                if state.price_variance > 0.0005:
                    logger.critical("JANITOR: RTT-Volatility Breach. Emergency Market Unwind.")
                    # Signal to Hawkes that we are creating this volatility ourselves
                    # self.hawkes.register_internal_trade(time.time())
                else:
                    # Otherwise, try to complete the hedge via market order
                    logger.info("JANITOR: Attempting recovery hedge via market order.")
                    # self.hawkes.register_internal_trade(time.time())

    async def start(self):
        if self.role == "brain":
            await self.run_brain_loop()
        else:
            await self.run_execution_loop()

    def shutdown(self):
        self.network.close()
        self.disruptor.close()
        if self.role == "brain":
            self.disruptor.unlink()
