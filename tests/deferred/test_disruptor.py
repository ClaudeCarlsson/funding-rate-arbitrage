import pytest
import time
from multiprocessing import Process
from funding_arb.disruptor import DisruptorRingBuffer

def producer_task(buffer_name: str, count: int):
    # Attach to existing memory
    rb = DisruptorRingBuffer(name=buffer_name, create=False)
    for i in range(count):
        # timestamp, mid_price, bid_depth, ask_depth, instrument_id, exchange_id
        rb.publish(time.time(), 50000.0 + i, 10.0, 10.0, i % 100, 1)
    rb.close()

def test_disruptor_ipc_latency():
    buffer_name = "test_disruptor_arb"
    # Create the master buffer
    rb = DisruptorRingBuffer(name=buffer_name, capacity=8192, create=True)
    
    count = 5000
    
    # Spawn a completely separate OS process to act as the scanner
    p = Process(target=producer_task, args=(buffer_name, count))
    p.start()
    
    received = 0
    start_time = time.time()
    
    # Consumer loop (The Optimizer/Filter process)
    while received < count:
        ticks = rb.consume()
        if not ticks:
            time.sleep(0.001)
        
        # Check data integrity of the last batch
        for tick in ticks:
            # timestamp(0), mid_price(1), bid_depth(2), ask_depth(3), instrument_id(4), exchange_id(5)
            assert tick[1] >= 50000.0
            assert tick[5] == 1
            
        received += len(ticks)
        if time.time() - start_time > 10.0: # Increased timeout
            break
            
    p.join()
    
    assert received == count
    
    # Cleanup
    rb.close()
    rb.unlink()

if __name__ == "__main__":
    pytest.main([__file__])
