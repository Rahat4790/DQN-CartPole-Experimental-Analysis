import torch
from dqn_network import DQN
from replay_memory import ReplayMemory


def test_network_output_shape():
    model = DQN(4, 2)
    output = model(torch.zeros(3, 4))
    assert output.shape == (3, 2)


def test_replay_memory_sample():
    memory = ReplayMemory(10)
    for i in range(5):
        memory.push(torch.tensor([[float(i)]]),
                    torch.tensor([[0]]),
                    None,
                    torch.tensor([1.0]))
    assert len(memory) == 5
    assert len(memory.sample(3)) == 3
