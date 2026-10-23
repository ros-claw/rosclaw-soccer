"""Separate recurrent student/teacher histories over one actually executed stream.

This interface produces advice, not authority, task success or qualified data.
The caller owns execution and must independently verify physical outcomes.
"""

from typing import Any

import torch

from rosclaw_soccer.rsi.consolidated_motor_adapter import ConsolidatedMotorAdapter
from rosclaw_soccer.rsi.dual_recurrent_motor import DualRecurrentMotor

MotorState = tuple[torch.Tensor, torch.Tensor]
PairState = tuple[MotorState, MotorState]
Motor = DualRecurrentMotor | ConsolidatedMotorAdapter


def paired_reference(motor: Motor) -> torch.nn.Module:
    """Explicit class admission; no duck-typed arbitrary policy wrappers."""
    if type(motor) is ConsolidatedMotorAdapter:
        return motor.memory.parent
    if type(motor) is DualRecurrentMotor:
        return motor.parent
    raise ValueError("explicit qualified dual motor or consolidated adapter required")


def _plastic_storage(motor: Motor) -> set[int]:
    if type(motor) is ConsolidatedMotorAdapter:
        params = (*motor.memory.plastic.parameters(), *motor.head.parameters())
    elif type(motor) is DualRecurrentMotor:
        params = tuple(motor.plastic.parameters())
    else:
        raise ValueError("explicit qualified motor required")
    return {p.untyped_storage().data_ptr() for p in params}


class SameStateMotorPair:
    """No hidden state in this object, no teacher state passed to the student."""

    def __init__(self, student: Motor, teacher: Motor) -> None:
        if (
            type(student) not in (DualRecurrentMotor, ConsolidatedMotorAdapter)
            or type(teacher) not in (DualRecurrentMotor, ConsolidatedMotorAdapter)
            or student is teacher
            or student.training
            or teacher.training
            or any(
                p.requires_grad or p.device.type != "cpu"
                for m in (student, teacher)
                for p in m.parameters()
            )
        ):
            raise ValueError("two separately admitted frozen CPU dual256 policies required")
        if _plastic_storage(student) & _plastic_storage(teacher):
            raise ValueError("learner and teacher plastic networks cannot alias")
        self.student, self.teacher = student, teacher

    def initial_state(self) -> PairState:
        return self.student.initial_state(), self.teacher.initial_state()

    def observe(
        self, observation: torch.Tensor, state: PairState
    ) -> tuple[torch.Tensor, torch.Tensor, PairState, dict[str, Any]]:
        if observation.shape != (1, 1, 160) or observation.device.type != "cpu":
            raise ValueError("one actually observed CPU160 control step required")
        if len(state) != 2 or any(len(s) != 2 for s in state):
            raise ValueError("separate student and teacher h/c required")
        student_storage = {s.untyped_storage().data_ptr() for s in state[0]}
        if any(s.untyped_storage().data_ptr() in student_storage for s in state[1]):
            raise ValueError("student cannot borrow the teacher recurrent state")
        with torch.no_grad():
            actual, after_s, se = self.student.forward_with_evidence(observation, state[0])
            advice, after_t, te = self.teacher.forward_with_evidence(observation, state[1])
        return (
            actual,
            advice,
            (after_s, after_t),
            dict(
                student=se,
                teacher=te,
                observation_source="ONE_ACTUAL_EXECUTED_STREAM_NOT_TWO_INDEPENDENT_ROLLOUTS",
                teacher_advice_is_executed=False,
                teacher_advice_physically_qualified=False,
                student_success_claim=False,
                hardware_authorized=False,
            ),
        )
