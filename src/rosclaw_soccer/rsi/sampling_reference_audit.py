"""Independent sampling audits: fixed complete mean, only bounded seed varies.

No producer factory or optimized producer preview is used. The original full
sampling constructor validates the prototype. Every audit deeply owns its
initial reference objects and uses the original stationary-noise function.
Physical parity and full-source authentication remain external prerequisites.
"""

import copy
import sys
from pathlib import Path
from typing import Any

import rosclaw.growth.canonical_json_snapshot as snapshot_module
from rosclaw.growth.canonical_json_snapshot import CanonicalJSONSnapshot
from rosclaw.growth.correlated_exploration import stationary_noise

from rosclaw_soccer.rsi.fixed_recurrent_reference_audit import _numeric_graph_hash
from rosclaw_soccer.rsi.recurrent_sampling_motor import CompiledRecurrentSamplingMotor
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


class SamplingReferenceAuditCompiler:
    """One fully validated immutable mean/template, independent fresh audits.

    The only permitted change to the original complete policy is the seed and
    the three hashes that the original preview derives from it. All ancestry,
    actor/critic, source, sampling law, action bounds and authority stay fixed.
    This option is not selected by any collector or auditor by default.
    """

    def __init__(self, policy: dict[str, Any]) -> None:
        if type(policy) is not dict or "recurrent_sampling_motor_proof" not in policy:
            raise ValueError("original complete recurrent sampling policy required")
        self._policy = CanonicalJSONSnapshot(policy)
        paths = list(Path(__file__).parent.glob("*.py"))
        paths += list(Path(snapshot_module.__file__).parent.glob("*.py"))
        paths += [Path(__file__).parents[1] / "sim/contracts.py"]
        self._pins = {str(path): hash_bytes(path.read_bytes()) for path in paths}
        self._prototype = CompiledRecurrentSamplingMotor(self._policy.restore())
        self._prototype_hash = _numeric_graph_hash(self._prototype)
        spatial_module = sys.modules.get("scipy.spatial._ckdtree")
        if spatial_module is not None:
            path = Path(str(spatial_module.__file__))
            self._pins[str(path)] = hash_bytes(path.read_bytes())
        self._initial_policy_hash = str(policy["policy_hash"])
        self._stable()

    def _stable(self) -> None:
        if (
            any(
                hash_bytes(Path(path).read_bytes()) != expected
                for path, expected in self._pins.items()
            )
            or self._policy.verify() != self._policy.content_hash
            or self._prototype._policy_hash != self._initial_policy_hash
            or _numeric_graph_hash(self._prototype) != self._prototype_hash
        ):
            raise ValueError("original sampling reference source, template or prototype changed")

    def _verified_policy(self, policy: dict[str, Any]) -> tuple[int, str]:
        self._stable()
        if (
            type(policy) is not dict
            or type(policy.get("step_motor_proof")) is not dict
            or type(policy["step_motor_proof"].get("model")) is not dict
        ):
            raise ValueError("complete original sampling policy required")
        owned = CanonicalJSONSnapshot(policy)
        # The complete owned canonical document already authenticates all
        # fields, not just the caller's policy_hash or the mean-model hash.
        # Exact repeats need no extra full restore, seed substitution or hash
        # encoding. Read the seed from the fully validated, graph-bound private
        # prototype, never from the caller's potentially mutable dictionary.
        # Keep both source/prototype stability checks, including the immutable
        # snapshot's byte digest; changed seeds still take the original path.
        if owned.content_hash == self._policy.content_hash:
            sampling = self._prototype._sampling
            if sampling is None:
                raise ValueError("original sampling reference law missing")
            seed = sampling.get("seed")
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError("bounded integer reference sampling seed required")
            self._stable()
            return seed, self._initial_policy_hash
        document = owned.restore()
        seed = document["step_motor_proof"]["model"].get("seed")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("bounded integer reference sampling seed required")
        expected = self._policy.restore()
        view = expected["step_motor_proof"]["model"]
        view["seed"] = seed
        view["model_hash"] = hash_json({k: v for k, v in view.items() if k != "model_hash"})
        expected["recurrent_sampling_motor_proof"]["sampling_model_hash"] = view["model_hash"]
        expected["policy_hash"] = hash_json(
            {k: v for k, v in expected.items() if k != "policy_hash"}
        )
        if owned.content_hash != CanonicalJSONSnapshot(expected).content_hash:
            raise ValueError("complete sampling mean, template, law or source changed")
        self._stable()
        return seed, str(expected["policy_hash"])

    def verify_policy(self, policy: dict[str, Any]) -> None:
        self._verified_policy(policy)

    def new_episode(self, policy: dict[str, Any]) -> CompiledRecurrentSamplingMotor:
        seed, policy_hash = self._verified_policy(policy)
        episode = copy.deepcopy(self._prototype)
        if _numeric_graph_hash(episode) != self._prototype_hash:
            raise ValueError("whole original initial sampling reference episode changed")
        if episode._sampling is None:
            raise ValueError("original sampling reference law missing")
        episode._sampling = dict(episode._sampling, seed=seed)
        episode._noise = stationary_noise(
            seed=seed, rho=episode._sampling["rho"], count=270, dimension=12, first_frame=30
        )
        episode._noise.flags.writeable = False
        episode._policy_hash = policy_hash
        self._stable()
        return episode

    def contract(self) -> dict[str, Any]:
        self._stable()
        return {
            "schema": "soccer.rsi.independent_sampling_reference_audit_compiler.v1",
            "initial_complete_policy_numeric_hash": self._policy.content_hash,
            "initial_policy_hash": self._initial_policy_hash,
            "initial_reference_graph_hash": self._prototype_hash,
            "source_pins": dict(self._pins),
            "only_seed_and_its_three_original_hashes_may_vary": True,
            "original_complete_sampling_constructor_used_at_allocation": True,
            "original_stationary_noise_function_used": True,
            "producer_factory_or_preview_reused": False,
            "all_initial_reference_objects_deeply_owned_per_episode": True,
            "physics_parity_requires_external_evidence": True,
            "actual_archive_parity_qualified_here": False,
            "physical_action_bounds_changed": False,
            "execution_ceiling": "OFFLINE_SIM_AUDIT_ONLY",
            "runtime_execution_authorized": False,
            "promotion_authorized": False,
            "hardware_authorized": False,
        }
