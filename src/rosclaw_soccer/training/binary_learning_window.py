"""Training-only metadata adapter; the policy never observes the extra bit."""

from typing import Any


def attach_binary_learning_window(learner: Any, *, observation_size: int) -> Any:
    """Append a validated sample-selection bit without changing policy outputs.

    Parameters are shared with ``learner``; no copy, initialization, optimizer,
    simulator, or runtime activation occurs. The wrapper state dict prefixes
    keys with ``learner.``. Export the learner state dict for the original motor
    contract; save the wrapper state dict for exact training resumption.
    The caller owns causal provenance of the bit and retention after masking.
    """
    import torch

    if (
        type(observation_size) is not int
        or not 1 <= observation_size < 4096
        or not isinstance(learner, torch.nn.Module)
        or not isinstance(getattr(learner, "logstd", None), torch.nn.Parameter)
    ):
        raise ValueError("bounded observation size and actor-critic with logstd required")

    class WindowedLearner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.learner = learner

        @property
        def logstd(self) -> Any:
            return self.learner.logstd

        def forward(self, observation: Any) -> Any:
            if (
                not isinstance(observation, torch.Tensor)
                or observation.ndim != 2
                or not 1 <= observation.shape[0] <= 65536
                or observation.shape[1] != observation_size + 1
                or observation.dtype not in (torch.float32, torch.float64)
                or not bool(torch.isfinite(observation).all())
                or not bool(((observation[:, -1] == 0) | (observation[:, -1] == 1)).all())
            ):
                raise ValueError(
                    "finite observations with an explicit binary learning bit required"
                )
            return self.learner(observation[:, :observation_size])

    return WindowedLearner()
