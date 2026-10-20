"""SIM-only explicit FSQ symbol classifier over a frozen recurrent body policy.

No Gaussian raw-density surrogate, straight-through rounding, teacher reference,
course ID, future state or motor authority. The frozen muscle is external.
"""

from typing import Any


def create_categorical_motion(parent: Any) -> Any:
    import torch
    from torch import nn

    if parent.training or any(p.requires_grad for p in parent.parameters()):
        raise ValueError("explicitly frozen recurrent kinematic parent required")

    class CategoricalMotion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.parent = parent
            self.head = nn.Sequential(nn.Linear(198, 128), nn.ELU(), nn.Linear(128, 64 * 32))
            output = self.head[-1]
            if not isinstance(output, nn.Linear):
                raise ValueError("explicit classifier output required")
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
            self.calls = 0

        def train(self, mode: bool = True) -> Any:
            super().train(mode)
            self.parent.eval()
            return self

        def context(self, observation: Any, hidden: Any = None) -> tuple[Any, Any, Any, Any]:
            with torch.no_grad():
                geometry = self.parent.geometry(observation)
                mean, _, value, next_hidden = self.parent.parameters_for_state(
                    observation, hidden, geometry=geometry
                )
                base = self.parent.parent.base
                norm = ((observation - base.observation_mean) / base.observation_std).clamp(-10, 10)
                geo = ((geometry - self.parent.geometry_mean) / self.parent.geometry_std).clamp(
                    -10, 10
                )
                context = torch.cat((norm, mean, geo), -1)
                token = (
                    torch.round(15.515501022338867 * torch.tanh(mean + 0.03223699703812599) - 0.5)
                    / 16
                )
                classes = (token * 16 + 16).long()
            return context, classes, value, next_hidden

        def logits(self, context: Any, classes: Any) -> Any:
            if (
                context.shape[-1] != 198
                or context.dtype != torch.float32
                or not torch.isfinite(context).all()
                or classes.shape != (*context.shape[:-1], 64)
                or classes.dtype != torch.int64
                or (classes < 0).any()
                or (classes > 31).any()
            ):
                raise ValueError(
                    "finite current context and exact original FSQ class prior required"
                )
            logits = self.head(context).reshape(*classes.shape, 32)
            prior = torch.nn.functional.one_hot(classes, num_classes=32).to(logits.dtype)
            result = logits + 4.0 * prior
            if not torch.isfinite(result).all():
                raise ValueError("nonfinite categorical action; no fallback invented")
            return result

        def forward(self, observation: Any, hidden: Any = None) -> tuple[Any, Any, Any]:
            context, classes, value, next_hidden = self.context(observation, hidden)
            chosen = self.logits(context, classes).argmax(-1)
            self.calls += 1
            decoded = (chosen.to(torch.float32) - 16.0) / 16.0
            mean = context[..., 108:172]
            original_token = (
                torch.round(15.515501022338867 * torch.tanh(mean + 0.03223699703812599) - 0.5)
                / 16.0
            )
            # Original class16 may be -0.0: preserve prior bits on unchanged classes.
            token = torch.where(chosen == classes, original_token, decoded)
            return token, value, next_hidden

        def network_call_counts(self) -> dict[str, int]:
            if self.calls != 300:
                raise ValueError("complete owned H300 inference required")
            # Frozen recurrent base, frozen kinematic branch, new symbol head.
            return dict(policy_invocations=300, intention_network_calls=900)

    return CategoricalMotion()
