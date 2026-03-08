import pytest
import asyncio
from funding_arb.raft import RaftNode, NodeState

@pytest.mark.asyncio
async def test_raft_quorum_success():
    # Simulate 3 Nodes: Brain (US), Binance (Tokyo), Bybit (Singapore)
    brain = RaftNode(node_id="brain_us", peers=["binance_tokyo", "bybit_singapore"])
    brain.state = NodeState.LEADER
    
    command = {
        "action": "EXECUTE", 
        "legs": ["Binance_Long_BTC", "Bybit_Short_BTC"],
        "h_inf_state": {"safe": True}
    }
    
    # Leader should propose, get 3/3 acks, and commit
    success = await brain.propose_execution(command)
    
    assert success is True
    assert brain.commit_index == 0
    assert len(brain.log) == 1

@pytest.mark.asyncio
async def test_raft_network_partition():
    # Simulate 3 Nodes
    brain = RaftNode(node_id="brain_us", peers=["binance_tokyo", "bybit_singapore"])
    brain.state = NodeState.LEADER
    
    # Intentionally sabotage the network send function to simulate a severed trans-pacific cable
    async def failing_send(peer, index, entry):
        return False
        
    brain.send_append_entries = failing_send
    
    command = {"action": "EXECUTE", "legs": ["Binance_Long", "Bybit_Short"]}
    
    # Leader will only get its own ack (1/3). Quorum requires 2.
    success = await brain.propose_execution(command)
    
    # The trade MUST NOT fire. Naked delta is prevented.
    assert success is False
    assert brain.commit_index == -1 

if __name__ == "__main__":
    pytest.main([__file__])
