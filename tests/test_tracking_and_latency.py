import pytest
import numpy as np
from funding_arb.tracking import OrderBookParticleFilter
from funding_arb.network_calculus import MinPlusLatencyVeto

def test_particle_filter_predict_update():
    pf = OrderBookParticleFilter(num_particles=500)
    
    # 1. Initialize
    pf.initialize(mid_price=50000.0, bid_depth=10.0, ask_depth=10.0, timestamp=1.0)
    state = pf.estimate_state()
    assert np.isclose(state.expected_mid_price, 50000.0)
    
    # 2. Predict step (data blackout)
    # Simulate high hawkes intensity
    pf.predict(current_time=1.5, hawkes_intensity=0.1, baseline_intensity=0.01)
    
    state_after_predict = pf.estimate_state()
    # Uncertainty (variance) should increase after a blackout
    assert state_after_predict.price_variance > 0
    assert state_after_predict.depth_variance > 0
    
    # 3. Update step (websocket reconnects with new price)
    pf.update(obs_mid=50050.0, obs_bid_depth=8.0, obs_ask_depth=12.0, timestamp=1.6)
    state_after_update = pf.estimate_state()
    
    # The expected mid price should have moved toward the new observation
    assert state_after_update.expected_mid_price > 50020.0 


def test_min_plus_latency_veto():
    veto = MinPlusLatencyVeto()
    
    # Record some pings
    for latency in [0.05, 0.04, 0.06, 0.05, 0.5]: # 0.5 is an outlier
        veto.record_ping(latency)
        
    R, T = veto.calculate_service_curve()
    
    # Since len is 5, 99th percentile is index 4 (the outlier 0.5)
    assert T == 0.5
    
    max_delay = veto.compute_max_delay(order_burst_size=2)
    # D = T + b/R = 0.5 + 2/10 = 0.7
    assert np.isclose(max_delay, 0.7)
    
    # If the opportunity only lasts 0.2s, we must veto
    assert veto.should_veto(expected_half_life=0.2, order_burst_size=2) is True
    
    # If the opportunity lasts 1.0s, we are safe to execute
    assert veto.should_veto(expected_half_life=1.0, order_burst_size=2) is False

if __name__ == "__main__":
    pytest.main([__file__])
