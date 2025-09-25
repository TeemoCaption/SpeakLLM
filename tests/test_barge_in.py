from __future__ import annotations

import torch

from llm.policy.duplex_controller import DuplexController


def test_duplex_controller_outputs_probabilities() -> None:
    controller = DuplexController()
    features = torch.zeros(1, controller.config.input_dim)
    logits = controller(features)
    probs = torch.softmax(logits, dim=-1)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
