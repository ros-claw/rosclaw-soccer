"""Fresh private receiving instances; never reset the world's foundation LSTM."""

from __future__ import annotations

from collections.abc import Callable

from rosclaw_soccer.providers.g1.admitted_receiver import AdmittedRecurrentReceiver
from rosclaw_soccer.providers.g1.recurrent_receiver import G1RecurrentReceiver
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_rearm import TeamMotorRearmContext


class CyclicRecurrentReceiver(AdmittedRecurrentReceiver):
    """A normal completion may yield a new bounded instance for a later lease.

    The world must explicitly enable and validate this transfer. A successor
    still has to observe an incoming ball before starting its own skill.
    """

    def __init__(
        self, factory: Callable[[], G1RecurrentReceiver], *, minimum_speed_mps: float = 0.4
    ) -> None:
        receiver = factory()
        if not isinstance(receiver, G1RecurrentReceiver) or (
            receiver._start_frame is not None or receiver._faulted
        ):
            raise ValueError("unused unfaulted private receiver factory required")
        super().__init__(receiver, minimum_speed_mps=minimum_speed_mps, retire_on_completion=True)
        self._factory = factory
        self.contract_hash = hash_json(
            dict(
                schema="soccer.cyclic_recurrent_receiver.v1",
                admitted_contract_hash=self.contract_hash,
                rearm="new_lease_after_normal_retirement_and_native_idle",
                activation_ceiling="SIM_ONLY",
            )
        )
        self._successor_issued = False

    def successor(self, context: TeamMotorRearmContext) -> CyclicRecurrentReceiver | None:
        if self.faulted or self.receiver._faulted:
            raise ValueError("receiver fault cannot be cleared by rearming")
        if not isinstance(context, TeamMotorRearmContext):
            raise ValueError("typed rearm context required")
        context.__post_init__()
        if context.agent_id != self.agent_id or context.contract_hash != self.contract_hash:
            raise ValueError("foreign rearm context")
        if not self.completed or context.retirement_frame != self._last_frame:
            raise ValueError("normal recorded completion required")
        if not context.eligible:
            return None
        if self._successor_issued:
            raise ValueError("receiver successor already issued")
        candidate = CyclicRecurrentReceiver(self._factory, minimum_speed_mps=self.minimum_speed_mps)
        if candidate.receiver is self.receiver or candidate.contract_hash != self.contract_hash:
            raise ValueError("receiver factory changed identity or reused live state")
        self._successor_issued = True
        return candidate
