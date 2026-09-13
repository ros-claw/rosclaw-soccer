"""Explicit, checkpointed feature scaling for experimental learning heads.

No fitting, data selection, simulator access or policy promotion occurs here.
The caller owns training-only statistics and their provenance.
"""

from typing import Any


def normalize_network_inputs(network: Any, mean: Any, scale: Any) -> Any:
    """Wrap a network with copied, non-trainable float32 normalization buffers.

    The checkpoint schema changes explicitly to network.*, mean, scale.
    Values are clipped to +/-10 after normalization, never written back into
    the raw observation or shared with a frozen parent policy.
    """
    import torch

    if not isinstance(network, torch.nn.Module):
        raise ValueError("Torch learning network required")
    if (
        not isinstance(mean, torch.Tensor)
        or not isinstance(scale, torch.Tensor)
        or mean.ndim != 1
        or scale.shape != mean.shape
        or not 1 <= len(mean) <= 4096
        or mean.device != scale.device
        or mean.dtype != torch.float32
        or scale.dtype != torch.float32
        or mean.requires_grad
        or scale.requires_grad
        or not bool(torch.isfinite(mean).all() and torch.isfinite(scale).all())
        or bool((mean.abs() > 1e6).any())
        or bool(((scale < 1e-6) | (scale > 1e6)).any())
    ):
        raise ValueError("explicit finite detached float32 normalization vectors required")

    class NormalizedNetwork(torch.nn.Module):
        mean: torch.Tensor
        scale: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.network = network
            self.register_buffer("mean", mean.detach().clone())
            self.register_buffer("scale", scale.detach().clone())

        def forward(self, observation: Any) -> Any:
            if (
                not isinstance(observation, torch.Tensor)
                or observation.ndim != 2
                or observation.shape[1] != len(self.mean)
                or not 1 <= len(observation) <= 65536
                or observation.dtype != torch.float32
                or observation.device != self.mean.device
                or not bool(torch.isfinite(observation).all())
                or bool((observation.abs() > 1e6).any())
                or not bool(torch.isfinite(self.mean).all() and torch.isfinite(self.scale).all())
                or bool((self.mean.abs() > 1e6).any())
                or bool(((self.scale < 1e-6) | (self.scale > 1e6)).any())
            ):
                raise ValueError("finite observation and valid fixed normalization required")
            return self.network(((observation - self.mean) / self.scale).clamp(-10, 10))

    return NormalizedNetwork()
