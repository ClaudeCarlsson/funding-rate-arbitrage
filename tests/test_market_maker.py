import pytest
import numpy as np
from funding_arb.market_maker import AvellanedaStoikovEngine

def test_as_engine_skew_logic():
    engine = AvellanedaStoikovEngine(risk_aversion=1.0, order_intensity=1.5, horizon_sec=100.0)
    
    mid_price = 50000.0
    volatility = 0.0001
    current_q = 0.0
    target_q = 1.0 # Brain Node wants us to buy 1 BTC
    
    # 1. No time elapsed, target is higher than current
    quotes = engine.calculate_quotes(mid_price, volatility, current_q, target_q)
    
    # Reservation price should skew HIGHER than mid-price (making bids more aggressive)
    assert quotes.reservation_price > mid_price
    # Both bid and ask move up relative to mid-price
    assert quotes.bid_price > mid_price - 10.0 # Standard half-spread is around this
    
    # 2. Reverse: We are long, target is to be flat (need to sell)
    current_q = 1.0
    target_q = 0.0
    quotes_sell = engine.calculate_quotes(mid_price, volatility, current_q, target_q)
    
    # Reservation price should skew LOWER than mid-price (making asks more aggressive)
    assert quotes_sell.reservation_price < mid_price
    assert quotes_sell.ask_price < mid_price + 10.0

def test_as_engine_spread_expansion():
    # In highly volatile markets, the spread must expand to protect the maker
    engine = AvellanedaStoikovEngine(risk_aversion=1.0)
    
    # Low vol
    q1 = engine.calculate_quotes(50000.0, 0.0001, 0.0, 0.0)
    # High vol
    q2 = engine.calculate_quotes(50000.0, 0.01, 0.0, 0.0)
    
    assert q2.spread > q1.spread

if __name__ == "__main__":
    pytest.main([__file__])
