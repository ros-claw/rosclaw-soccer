import random

import numpy as np
import pytest
import torch

from scripts.rsi_incoming_checkpoint import digest_file, restore_checkpoint, save_checkpoint


def test_episode_resume_reproduces_optimizer_and_random_streams(tmp_path):
    torch.manual_seed(17)
    np.random.seed(18)
    random.seed(19)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def update(net, opt):
        x = torch.randn(3, 2) + np.random.random() + random.random()
        opt.zero_grad()
        net(x).square().mean().backward()
        opt.step()
        return x

    update(model, optimizer)
    trace = tmp_path / "trace.npz"
    trace.write_bytes(b"complete trace")
    checkpoint = tmp_path / "episode.pt"
    commitment = {"source_head": "source", "input": "sha256:data"}
    checkpoint_hash = save_checkpoint(
        checkpoint,
        policy=model,
        optimizer=optimizer,
        commitment=commitment,
        episode=0,
        updates=1,
        stats=[{"raw_path": str(trace), "raw_hash": digest_file(trace)}],
        histories=[np.array([1, 2])],
    )
    expected_input = update(model, optimizer)
    expected_parameters = {k: v.clone() for k, v in model.state_dict().items()}
    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
    state = restore_checkpoint(
        checkpoint,
        checkpoint_hash,
        policy=restored,
        optimizer=restored_optimizer,
        commitment=commitment,
    )
    assert state["next_episode"] == 1
    assert torch.equal(update(restored, restored_optimizer), expected_input)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in expected_parameters.items())
    with pytest.raises(ValueError, match="commitment"):
        restore_checkpoint(
            checkpoint,
            checkpoint_hash,
            policy=restored,
            optimizer=restored_optimizer,
            commitment={"source_head": "wrong"},
        )
    trace.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="trajectory"):
        restore_checkpoint(
            checkpoint,
            checkpoint_hash,
            policy=restored,
            optimizer=restored_optimizer,
            commitment=commitment,
        )
