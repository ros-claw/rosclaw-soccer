"""Clock-checked sample holding for simulation-only temporal policy experiments.

This owns no actuator, motor mapping, physics state or learning authority.
Collectors must distinguish fresh random draws from actually applied samples,
and optimize only decisions whose samples can affect the environment.
"""

from typing import Any


class HeldPolicySample:
    """Hold a detached policy draw for a bounded number of control ticks.

    New instance per declared episode. A malformed input or broken clock is
    latched; callers cannot silently repair continuity by retrying a later tick.
    Closing a physical action window remains the caller's separate obligation.
    """

    def __init__(self, period: int = 1) -> None:
        if type(period) is not int or not 1 <= period <= 10:
            raise ValueError("integer hold period between one and ten ticks required")
        self._period = period
        self._next_tick = 0
        self._held: Any = None
        self._invalid = False

    def advance(self, draw: Any, *, tick: int) -> tuple[Any, bool]:
        """Return a private value copy and whether this tick starts a decision."""
        import torch

        if self._invalid:
            raise RuntimeError("policy sample clock is invalid")
        try:
            if type(tick) is not int or tick != self._next_tick:
                raise ValueError("consecutive policy control ticks required")
            if (
                not isinstance(draw, torch.Tensor)
                or draw.ndim != 2
                or not 1 <= draw.shape[0] <= 4096
                or not 1 <= draw.shape[1] <= 64
                or draw.dtype not in (torch.float32, torch.float64)
                or draw.layout != torch.strided
                or draw.requires_grad
                or not bool(torch.isfinite(draw).all())
                or bool((draw.abs() > 1e6).any())
            ):
                raise ValueError("bounded detached batched policy draw required")
            if self._held is not None and (
                draw.shape != self._held.shape
                or draw.dtype != self._held.dtype
                or draw.device != self._held.device
            ):
                raise ValueError("policy sample batch contract changed")
            decision = tick % self._period == 0
            if decision:
                self._held = draw.clone()
            result = self._held.clone()
        except Exception:
            self._invalid = True
            raise
        self._next_tick += 1
        return result, decision
