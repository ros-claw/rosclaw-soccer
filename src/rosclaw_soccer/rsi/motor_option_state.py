"""Explicit per-player option memory; never hide commitment in model globals."""

import re
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MotorOptionState:
    recurrent: tuple[torch.Tensor, torch.Tensor]
    generation_hash: str
    option: int | None = None
    controls: int = 0

    def validate(self) -> None:
        if (
            type(self.generation_hash) is not str
            or re.fullmatch(r"sha256:[a-f0-9]{64}", self.generation_hash) is None
            or type(self.controls) is not int
            or not 0 <= self.controls <= 512
            or (
                self.option is not None
                and (type(self.option) is not int or not 0 <= self.option <= 3)
            )
            or (self.option is None) != (self.controls == 0)
            or type(self.recurrent) is not tuple
            or len(self.recurrent) != 2
            or any(
                not isinstance(x, torch.Tensor)
                or x.shape != (2, 1, 256)
                or x.dtype != torch.float32
                or x.device.type != "cpu"
                or not torch.isfinite(x).all()
                for x in self.recurrent
            )
            or self.recurrent[0].untyped_storage().data_ptr()
            == self.recurrent[1].untyped_storage().data_ptr()
        ):
            raise ValueError("explicit finite private CPU option+dual256 memory required")

    def clone(self) -> "MotorOptionState":
        self.validate()
        return MotorOptionState(
            (self.recurrent[0].detach().clone(), self.recurrent[1].detach().clone()),
            self.generation_hash,
            self.option,
            self.controls,
        )
