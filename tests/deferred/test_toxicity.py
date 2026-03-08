import pytest
from funding_arb.toxicity import VPINDetector, Trade

def test_vpin_calculation():
    # Use small bucket volumes for testing
    detector = VPINDetector(bucket_volume=10.0, num_buckets=5)
    
    # 1. Balanced trading (low toxicity)
    for _ in range(5):
        detector.process_trade(Trade(1.0, 50000.0, 5.0, 'buy'))
        detector.process_trade(Trade(1.1, 50000.0, 5.0, 'sell'))
        
    vpin_low = detector.calculate_vpin()
    assert vpin_low < 0.2
    
    # 2. Toxic directional trading (high toxicity)
    for _ in range(5):
        # Pure buys
        detector.process_trade(Trade(2.0, 50001.0, 10.0, 'buy'))
        
    vpin_high = detector.calculate_vpin()
    assert vpin_high > 0.8
    assert detector.is_toxic(threshold=0.7) is True

if __name__ == "__main__":
    pytest.main([__file__])
