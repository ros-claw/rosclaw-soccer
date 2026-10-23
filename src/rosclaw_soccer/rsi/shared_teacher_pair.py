"""Two private SIM motor histories on ONE actual body observation stream.

Both policies receive the actual executed action on commit. Their h/c are
never exchanged; an expert proposal is advice, not qualified DAgger data.
The caller alone selects an action and advances the shared native world.
"""

import numpy as np
import torch

from rosclaw_soccer.rsi.same_state_motor_pair import SameStateMotorPair, paired_reference
from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer


class SharedTeacherPair:
    def __init__(
        self, student: SharedNativeNeuralPlayer, teacher: SharedNativeNeuralPlayer
    ) -> None:
        if (
            type(student) is not SharedNativeNeuralPlayer
            or type(teacher) is not SharedNativeNeuralPlayer
            or student is teacher
            or student.world is not teacher.world
            or student.prefix != teacher.prefix
            or student.clip is not teacher.clip
            or student.build_observation is not teacher.build_observation
            or student.torso_index != teacher.torso_index
            or student.world.control_steps != 0
            or any(p.pending is not None or p.decisions or p.frame for p in (student, teacher))
            or any(
                not np.array_equal(getattr(student, k), getattr(teacher, k))
                for k in ("goal", "offset", "scale", "order")
            )
        ):
            raise ValueError("two fresh private policies on the exact SAME body/task/ABI required")
        SameStateMotorPair(student.actor, teacher.actor)
        s, t = (
            paired_reference(student.actor).state_dict(),
            paired_reference(teacher.actor).state_dict(),
        )
        if s.keys() != t.keys() or any(not torch.equal(v, t[k]) for k, v in s.items()):
            raise ValueError("identical qualified frozen reference memory required")
        if any(
            a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
            for a in student.state
            for b in teacher.state
        ):
            raise ValueError("student and teacher cannot share hidden state storage")
        self.student, self.teacher = student, teacher
        self.pending = False
        self.faulted = False

    @property
    def phase(self) -> str:
        if self.student.phase != self.teacher.phase or self.student.frame != self.teacher.frame:
            raise ValueError("same actual reference phase required")
        return self.student.phase

    def begin_standing(self) -> None:
        if self.faulted or self.pending or self.phase != "MOTION_COMPLETE":
            raise ValueError("complete healthy actual motion boundary required")
        self.student.begin_standing()
        self.teacher.begin_standing()

    def propose(
        self,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
        if self.faulted or self.pending:
            raise ValueError("one live private pair proposal required")
        try:
            a, sa = self.student.propose(with_reference_evidence=True)
            b, ta = self.teacher.propose(with_reference_evidence=True)
            if any(not np.array_equal(sa[k], ta[k]) for k in ("NN_obs", "NN_reference_nominal")):
                raise ValueError(
                    "advice must use SAME actual observation/reference, not a teacher rollout"
                )
            self.pending = True
            return a, sa, b, ta
        except Exception:
            self.faulted = True
            raise

    def commit(self, actual_action: np.ndarray) -> None:
        if self.faulted or not self.pending:
            raise ValueError("own pending pair and one ACTUAL executed action required")
        try:
            self.student.commit(executed_action=actual_action)
            self.teacher.commit(executed_action=actual_action)
            self.pending = False
        except Exception:
            self.faulted = True
            raise
