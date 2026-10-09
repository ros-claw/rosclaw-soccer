"""Private SIM numeric episodes; independently qualified physics still needed.

Original complete model/preview/decoder validation runs at allocation. Only
fixed read-only numeric parameters are reused; all causal episode state is
fresh. This factory is NOT selected by any existing collector or native CLI.
"""

import copy
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.canonical_json_snapshot as snapshot_module
from rosclaw.growth.correlated_exploration import stationary_noise

from rosclaw_soccer.rsi.fixed_recurrent_reference_audit import _numeric_graph_hash
from rosclaw_soccer.rsi.owned_recurrent_sampling_preview import OwnedRecurrentSamplingPreview
from rosclaw_soccer.rsi.recurrent_sampling_motor import CompiledRecurrentSamplingMotor
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _contract(mean_hash: str, pins: dict[str, str]) -> dict[str, Any]:
    return dict(
        schema="soccer.rsi.private_recurrent_sampling_episode_factory.v1",
        complete_canonical_mean_hash=mean_hash,
        source_pins=dict(pins),
        complete_original_model_and_decoder_validation_at_allocation=True,
        complete_original_preview_verification_at_each_bind=True,
        independent_contact_history=True,
        independent_recurrent_hidden_state=True,
        independent_stationary_ar_draws=True,
        actor_or_critic_weights_changed=False,
        physical_action_bounds_changed=False,
        physics_parity_requires_external_evidence=True,
        native_transport_qualification_performed=False,
        activation_ceiling="SIM_ONLY",
        runtime_execution_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )


def validate_compilation_contract(
    value: Any,
    policy: dict[str, Any],
    *,
    source_roots: tuple[Path, Path] | None = None,
) -> None:
    """Fixed source/identity declaration only; never physical or policy approval.

    The independent auditor must still use the original complete decoder and
    reconstruct every frame and the actual World/actuator-control dynamics.
    Never follow declaration-provided paths when checking this declaration.
    An offline reviewer may explicitly select independently pinned producer
    Soccer/Core roots for a historical archive. This validates producer
    provenance, not the currently loaded independent decoder or physics.
    """
    if (
        type(value) is not dict
        or type(policy) is not dict
        or type(policy.get("step_motor_proof")) is not dict
        or type(policy["step_motor_proof"].get("model")) is not dict
    ):
        raise ValueError("complete recurrent sampling compilation identity required")
    view = policy["step_motor_proof"]["model"]
    if (
        view.get("schema") != "soccer.rsi.recurrent_motor_sampling.v1"
        or type(view.get("mean_model")) is not dict
        or "recurrent_sampling_motor_proof" not in policy
        or policy.get("policy_hash")
        != hash_json({k: v for k, v in policy.items() if k != "policy_hash"})
    ):
        raise ValueError("complete original recurrent sampling preview identity required")
    if source_roots is None:
        paths = list(Path(__file__).parent.glob("*.py"))
        paths += list(Path(snapshot_module.__file__).parent.glob("*.py"))
        paths += [Path(__file__).parents[1] / "sim/contracts.py"]
    else:
        if (
            type(source_roots) is not tuple
            or len(source_roots) != 2
            or any(not isinstance(root, Path) or not root.is_absolute() for root in source_roots)
        ):
            raise ValueError("two explicit absolute reviewer-selected source roots required")
        soccer, core = source_roots
        folders = (soccer / "src/rosclaw_soccer/rsi", core / "src/rosclaw/growth")
        if any(not folder.is_dir() or folder.is_symlink() for folder in folders):
            raise ValueError("complete independently pinned producer source folders required")
        paths = [path for folder in folders for path in folder.glob("*.py")]
        paths += [soccer / "src/rosclaw_soccer/sim/contracts.py"]
        if any(not path.is_file() or path.is_symlink() for path in paths):
            raise ValueError("ordinary complete producer source files required")
    pins = {str(p): hash_bytes(p.read_bytes()) for p in paths}
    expected = _contract(hash_json(view["mean_model"]), pins)
    if hash_json(value) != hash_json(expected):
        raise ValueError("complete fixed mean/source/non-authorizing compilation contract required")


class RecurrentSamplingEpisodeFactory:
    """Fixed full student plus separately owned body history and AR draws."""

    def __init__(self, mean_model: dict[str, Any]) -> None:
        self._preview = OwnedRecurrentSamplingPreview(mean_model)
        view = self._preview.sampling_view(seed=0)
        policy = self._preview.preview(view)
        # The original decoder checks the entire original preview and builds
        # the original allowlisted numeric objects, not an external callback.
        self._prototype = CompiledRecurrentSamplingMotor(policy)
        self._prototype_hash = _numeric_graph_hash(self._prototype)
        self._mean_hash = hash_json(view["mean_model"])
        self._initial_policy_hash = str(policy["policy_hash"])
        self._stable()

    def _stable(self) -> None:
        self._preview._stable()
        if (
            self._prototype._policy_hash != self._initial_policy_hash
            or self._mean_hash != self._preview._mean_hash
            or _numeric_graph_hash(self._prototype) != self._prototype_hash
        ):
            raise ValueError("fixed original recurrent prototype commitment changed")

    def preview(self, view: dict[str, Any]) -> dict[str, Any]:
        self._stable()
        result = self._preview.preview(view)
        self._stable()
        return result

    def sampling_view(self, *, seed: int) -> dict[str, Any]:
        self._stable()
        result = self._preview.sampling_view(seed=seed)
        self._stable()
        return result

    def bind(self, policy: dict[str, Any]) -> CompiledRecurrentSamplingMotor:
        self._stable()
        view = self._preview.validate_preview(policy)
        decoder = copy.copy(self._prototype)
        behavior = copy.copy(self._prototype._behavior)
        behavior._layers = list(self._prototype._behavior._layers)
        behavior._parent = copy.copy(self._prototype._behavior._parent)
        behavior._parent._layers = list(self._prototype._behavior._parent._layers)
        behavior._parent._residual_layers = list(self._prototype._behavior._parent._residual_layers)
        behavior._parent._warm = copy.copy(self._prototype._behavior._parent._warm)
        behavior._parent._warm.layers = list(self._prototype._behavior._parent._warm.layers)
        behavior._parent._warm.sampling = copy.deepcopy(
            self._prototype._behavior._parent._warm.sampling
        )
        behavior._parent._memory = ContactPhaseMemory()
        behavior._memory = ContactPhaseMemory()
        behavior._sampling = None
        decoder._behavior, decoder._parent = behavior, behavior._parent
        decoder._recurrent = self._prototype._recurrent.new_episode()
        decoder._memory = ContactPhaseMemory()
        decoder._active_frame = None
        decoder._sampling = {k: view[k] for k in ("seed", "std_raw", "rho")}
        decoder._noise = stationary_noise(
            seed=view["seed"], rho=view["rho"], count=270, dimension=12, first_frame=30
        )
        decoder._noise.flags.writeable = False
        decoder._policy_hash = policy["policy_hash"]
        decoder._last_mean = np.zeros(12)
        decoder._last_draw = np.zeros(12)
        decoder._last_log_probability = 0.0
        decoder._last_sampled = False
        self._stable()
        return decoder

    def contract(self) -> dict[str, Any]:
        self._stable()
        return _contract(self._mean_hash, self._preview._pins)
