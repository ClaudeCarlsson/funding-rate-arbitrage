import pytest
import asyncio
import time
import numpy as np
from unittest.mock import MagicMock, patch
from funding_arb.hft_node import HFTNode
from funding_arb.aeron_udp import AeronUDPNode
from funding_arb.prediction import HawkesPredictor

@pytest.mark.asyncio
async def test_janitor_desync_recovery():
    """
    Stress test: Tokyo fills BTC, Singapore fails.
    The Janitor must detect the 1.0 BTC drift within one RTT and trigger recovery.
    """
    node = HFTNode(role="brain", node_id="brain_test", peers=["tokyo", "singapore"])
    node.particle_filter.initialize(50000.0, 100.0, 100.0, time.time()) # Deep book
    
    mock_msg = {
        "type": "LEG_FILL",
        "instrument": "BTC/USDT",
        "exchange": "binance",
        "amount": 0.0 # Singapore filled 0
    }
    
    with patch.object(node.network, "poll", return_value=mock_msg):
        with patch("funding_arb.hft_node.logger") as mock_logger:
            await node._janitor_reconciliation()
            
            drift_logged = any("Delta Drift Detected" in call.args[0] for call in mock_logger.error.call_args_list)
            assert drift_logged is True
            # Because book is deep (100) and drift is 1.0, impact is 1%. No veto.
            attempt_hedge = any("Attempting recovery hedge" in call.args[0] for call in mock_logger.info.call_args_list)
            assert attempt_hedge is True

@pytest.mark.asyncio
async def test_janitor_market_impact_veto():
    """
    Stress test: Desync happens during thin liquidity.
    The Janitor must refuse to fire a market order to prevent a flash crash.
    """
    node = HFTNode(role="brain", node_id="brain_test", peers=["tokyo", "singapore"])
    
    # Very thin book. Only 2.0 BTC available.
    node.particle_filter.initialize(50000.0, 2.0, 2.0, time.time())
    
    mock_msg = {"type": "LEG_FILL", "instrument": "BTC/USDT", "amount": 0.0}
    
    with patch.object(node.network, "poll", return_value=mock_msg):
        with patch("funding_arb.hft_node.logger") as mock_logger:
            await node._janitor_reconciliation()
            
            # Drift is 1.0. Book is 2.0. Impact is 50%. This exceeds 20% limit.
            veto_logged = any("Market Impact Veto" in call.args[0] for call in mock_logger.critical.call_args_list)
            assert veto_logged is True

def test_internal_signal_blanking():
    """
    Ensure the Hawkes Process ignores our own internal unwinds.
    """
    hp = HawkesPredictor(baseline=0.01, alpha=0.5, beta=0.8)
    
    current_time = 1.0
    # A massive trade prints on the tape
    hp.events.append(0.9)
    
    # If it was an external trade, intensity spikes
    intensity_external = hp.compute_intensity(current_time)
    assert intensity_external > 0.4
    
    # If the Janitor tells the Hawkes model "that was me"
    hp.register_internal_trade(0.9)
    intensity_internal = hp.compute_intensity(current_time)
    
    # The intensity should collapse back to baseline because we subtracted our own impact
    assert np.isclose(intensity_internal, 0.01)

if __name__ == "__main__":
    pytest.main([__file__])
