import pytest
from funding_arb.queue_simulator import FIFOQueueSimulator

def test_fifo_queue_simulator_decay_and_fill():
    # Cancellation hazard rate of 0.1 (10% decay per second based on Q)
    simulator = FIFOQueueSimulator(cancellation_hazard_rate=0.1)
    
    # We place a buy order for 5.0 units.
    # There is 100.0 units of volume ahead of us in the queue.
    simulator.place_order(
        order_id="order_1",
        side="buy",
        price=50000.0,
        amount=5.0,
        current_book_depth_at_price=100.0
    )
    
    # Tick 1: 1 second passes. Aggressive sell volume hits the bid (10 units)
    # dQ = (10 + 0.1 * 100) * 1.0 = 20
    # Q becomes 80
    fills = simulator.process_tick(dt=1.0, aggressive_buy_volume=0.0, aggressive_sell_volume=10.0)
    
    assert len(fills) == 0
    assert simulator.active_orders["order_1"].queue_position == 80.0
    assert simulator.active_orders["order_1"].filled_amount == 0.0
    
    # Tick 2: A massive market sell order hits (100 units).
    # dQ = (100 + 0.1 * 80) * 1.0 = 108
    # Q would drop to -28. We are at the front.
    # We should get a full fill (5.0 units).
    fills = simulator.process_tick(dt=1.0, aggressive_buy_volume=0.0, aggressive_sell_volume=100.0)
    
    assert len(fills) == 1
    assert fills[0] == ("order_1", 5.0)
    
    order = simulator.active_orders["order_1"]
    assert order.filled_amount == 5.0
    assert order.queue_position == 0.0
    assert order.is_active is False

if __name__ == "__main__":
    pytest.main([__file__])
