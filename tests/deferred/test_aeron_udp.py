import pytest
import time
from multiprocessing import Process, Queue
from funding_arb.aeron_udp import AeronUDPNode, UDPConfig

def udp_listener_task(queue: Queue, config: UDPConfig):
    """A background process simulating an Execution Node in Tokyo/Singapore."""
    listener = AeronUDPNode(config=config, is_publisher=False)
    
    start_time = time.time()
    while time.time() - start_time < 2.0:
        msg = listener.poll()
        if msg is not None:
            queue.put(msg)
            break
        time.sleep(0.01)
        
    listener.close()

def test_aeron_udp_multicast():
    # Use a custom port to avoid conflicts
    config = UDPConfig(port=10001)
    
    # We use a multiprocessing queue just to get the result back to the test runner
    queue = Queue()
    
    # 1. Start the listener (Execution Node)
    p = Process(target=udp_listener_task, args=(queue, config))
    p.start()
    
    # Give the OS a moment to bind the socket
    time.sleep(0.1)
    
    # 2. Start the publisher (Brain Node)
    publisher = AeronUDPNode(config=config, is_publisher=True)
    
    # Multicast a Raft command
    test_command = {"action": "EXECUTE", "legs": ["Spot", "Perp"]}
    publisher.publish_raft_log(term=1, index=42, command=test_command)
    
    # 3. Wait for the listener to receive it
    p.join(timeout=2.0)
    
    # Verify the message traversed the network stack
    assert not queue.empty()
    received_msg = queue.get()
    
    assert received_msg["type"] == "RAFT_APPEND"
    assert received_msg["term"] == 1
    assert received_msg["index"] == 42
    assert received_msg["command"]["action"] == "EXECUTE"
    
    publisher.close()

if __name__ == "__main__":
    pytest.main([__file__])
