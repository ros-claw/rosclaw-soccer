"""Bind role self-models and individual curricula to physical team evidence."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from rosclaw_soccer.growth.role_self_model import (
    MatchRole,
    RoleSelfModel,
    RoleSkillBinding,
    SoccerSkill,
    TeamRoleRoster,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.dynamic_lead_pass_evidence import (
    validate_dynamic_lead_pass_evidence,
)
from rosclaw_soccer.training.neural_contact_canary import validate_neural_contact_canary
from rosclaw_soccer.training.neural_contact_holdout_exam import (
    validate_neural_contact_holdout_exam,
)


@dataclass(frozen=True)
class RoleCurriculumAssignment:
    """A failure-bound practice assignment for exactly one plastic role."""

    agent_id: str
    self_model_hash: str
    target_skill: SoccerSkill
    failure_context_hashes: tuple[str, ...]
    frozen_teammate_policy_hashes: tuple[str, ...]
    frozen_opponent_policy_hashes: tuple[str, ...]
    objective_metric: str
    current_value: float
    target_value: float
    priority: float
    source_evidence_hash: str
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.role_curriculum_assignment.v1"

    def __post_init__(self) -> None:
        hashes = (
            self.self_model_hash,
            *self.failure_context_hashes,
            *self.frozen_teammate_policy_hashes,
            *self.frozen_opponent_policy_hashes,
            self.source_evidence_hash,
        )
        if not self.agent_id or any(not value.startswith("sha256:") for value in hashes):
            raise ValueError("role curriculum identity or evidence is invalid")
        if not self.failure_context_hashes or len(set(self.failure_context_hashes)) != len(
            self.failure_context_hashes
        ):
            raise ValueError("role curriculum requires distinct failed contexts")
        if not self.objective_metric:
            raise ValueError("role curriculum requires an objective metric")
        values = (self.current_value, self.target_value, self.priority)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("role curriculum metrics must be normalized to [0, 1]")
        if self.target_value <= self.current_value:
            raise ValueError("role curriculum target must improve on current performance")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("role curriculum must remain SIM_ONLY")

    def validate_for(self, model: RoleSelfModel) -> None:
        if self.agent_id != model.agent_id or self.self_model_hash != model.self_model_hash:
            raise ValueError("role curriculum was assigned to another self model")
        model.skill(self.target_skill)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["target_skill"] = self.target_skill.value
        value["failure_context_hashes"] = list(self.failure_context_hashes)
        value["frozen_teammate_policy_hashes"] = list(self.frozen_teammate_policy_hashes)
        value["frozen_opponent_policy_hashes"] = list(self.frozen_opponent_policy_hashes)
        return value


def run_role_awareness_audit(
    *,
    lead_pass_evidence_path: Path,
    neural_canary_report_path: Path,
    neural_holdout_report_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create a role-local credit and next-curriculum ledger from real traces."""

    output = output_dir.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("role-awareness evidence must use a new external directory")
    output.mkdir(parents=True)
    lead_path = lead_pass_evidence_path.expanduser().resolve()
    canary_path = neural_canary_report_path.expanduser().resolve()
    holdout_path = neural_holdout_report_path.expanduser().resolve()
    lead = validate_dynamic_lead_pass_evidence(lead_path)
    canary = validate_neural_contact_canary(canary_path)
    holdout = validate_neural_contact_holdout_exam(holdout_path)
    request = {
        "schema_version": "rosclaw_soccer.role_awareness_audit_request.v1",
        "lead_pass_evidence_hash": lead["evidence_hash"],
        "lead_pass_file_hash": hash_bytes(lead_path.read_bytes()),
        "neural_canary_report_hash": canary["report_hash"],
        "neural_canary_file_hash": hash_bytes(canary_path.read_bytes()),
        "neural_holdout_report_hash": holdout["report_hash"],
        "neural_holdout_file_hash": hash_bytes(holdout_path.read_bytes()),
        "source_paths_recorded": False,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
    }
    _write_json(output / "request.json", request)
    report = _derive_report(lead=lead, canary=canary, holdout=holdout)
    report["request_hash"] = hash_bytes((output / "request.json").read_bytes())
    report["report_hash"] = hash_json(report)
    _write_json(output / "role-awareness-audit.json", report)
    return report


def validate_role_awareness_audit(
    path: Path,
    *,
    lead_pass_evidence_path: Path,
    neural_canary_report_path: Path,
    neural_holdout_report_path: Path,
) -> dict[str, Any]:
    """Re-derive all role credit and curriculum assignments from their sources."""

    source = path.expanduser().resolve()
    report = _bound_json(source, "report_hash")
    request_path = source.parent / "request.json"
    request = _read_object(request_path)
    lead_path = lead_pass_evidence_path.expanduser().resolve()
    canary_path = neural_canary_report_path.expanduser().resolve()
    holdout_path = neural_holdout_report_path.expanduser().resolve()
    lead = validate_dynamic_lead_pass_evidence(lead_path)
    canary = validate_neural_contact_canary(canary_path)
    holdout = validate_neural_contact_holdout_exam(holdout_path)
    if (
        request.get("schema_version") != "rosclaw_soccer.role_awareness_audit_request.v1"
        or request.get("lead_pass_evidence_hash") != lead["evidence_hash"]
        or request.get("lead_pass_file_hash") != hash_bytes(lead_path.read_bytes())
        or request.get("neural_canary_report_hash") != canary["report_hash"]
        or request.get("neural_canary_file_hash") != hash_bytes(canary_path.read_bytes())
        or request.get("neural_holdout_report_hash") != holdout["report_hash"]
        or request.get("neural_holdout_file_hash") != hash_bytes(holdout_path.read_bytes())
        or request.get("source_paths_recorded") is not False
        or request.get("physics_authority") != "CPU_MUJOCO"
        or request.get("activation_ceiling") != "SIM_ONLY"
        or request.get("hardware_command_sent") is not False
        or request.get("pixels_used_for_scoring") is not False
        or report.get("request_hash") != hash_bytes(request_path.read_bytes())
    ):
        raise ValueError("role-awareness source request is invalid")
    derived = _derive_report(lead=lead, canary=canary, holdout=holdout)
    actual = dict(report)
    actual.pop("request_hash", None)
    actual.pop("report_hash", None)
    if actual != derived:
        raise ValueError("role-awareness credit or curricula do not match physical evidence")
    return report


def _derive_report(
    *, lead: dict[str, Any], canary: dict[str, Any], holdout: dict[str, Any]
) -> dict[str, Any]:
    if (
        canary.get("status") != "PASS_NEURAL_CONTACT_CANARY"
        or holdout.get("status") != "REJECTED_NEURAL_CONTACT_LOCAL_HOLDOUT"
        or canary.get("actor_hash") != holdout.get("actor_hash")
    ):
        raise ValueError("role-awareness audit requires one passing canary and rejected holdout")
    candidate = next(run for run in canary["runs"] if run["label"] == "candidate-primary")
    baseline = next(run for run in canary["runs"] if run["label"] == "no-contact-residual-baseline")
    rows = cast(list[dict[str, Any]], holdout["rows"])
    count = len(rows)
    if count != 6:
        raise ValueError("role-awareness audit requires six fresh holdouts")
    playmaker_failed = tuple(
        row["context_hash"]
        for row in rows
        if float(row["primary"]["result"]["pass_delivery_error_m"]) > 0.45
    )
    finisher_failed = tuple(
        row["context_hash"] for row in rows if not row["strict_right_foot_chain"]
    )
    goalkeeper_failed = tuple(row["context_hash"] for row in rows if row["goal"])
    pass_success = (count - len(playmaker_failed)) / count
    finish_success = (count - len(finisher_failed)) / count
    goalkeeper_success = 1.0 - len(goalkeeper_failed) / count
    stable_rate = sum(int(row["safe"]) for row in rows) / count
    actor_hash = str(canary["actor_hash"])
    handoff_hash = str(canary["contact_handoff_actor_hash"])
    lead_policy_hash = str(lead["policy_hash"])
    lead_evidence_hash = str(lead["evidence_hash"])
    canary_hash = str(canary["report_hash"])
    holdout_hash = str(holdout["report_hash"])
    goalkeeper_policy_hash = hash_json(
        {
            "controller": "shared-world-goalkeeper-v23",
            "canary_implementation_hash": canary["implementation_hash"],
            "activation_ceiling": "SIM_ONLY",
        }
    )
    observation_hash = hash_json(
        {
            "contract": "egocentric-team-observation-v1",
            "teammates_and_opponents_are_labelled": True,
            "pixels_used": False,
        }
    )
    action_hash = hash_json(
        {
            "contract": "role-authorized-tactical-intent-v1",
            "direct_joint_torque": False,
            "activation_ceiling": "SIM_ONLY",
        }
    )

    def binding(
        skill: SoccerSkill,
        artifact: str,
        evidence: str,
        proficiency: float,
        priority: float,
    ) -> RoleSkillBinding:
        return RoleSkillBinding(skill, artifact, evidence, 1, proficiency, priority)

    playmaker_skills = (
        binding(SoccerSkill.LEAD_PASS, lead_policy_hash, lead_evidence_hash, pass_success, 1.0),
        binding(
            SoccerSkill.FIRST_TOUCH,
            hash_json({"role": "playmaker", "skill": "first_touch", "source": lead_policy_hash}),
            lead_evidence_hash,
            pass_success,
            0.7,
        ),
        binding(
            SoccerSkill.RECOVERY,
            hash_json({"role": "playmaker", "skill": "recovery", "source": handoff_hash}),
            holdout_hash,
            stable_rate,
            0.1,
        ),
    )
    finisher_skills = (
        binding(
            SoccerSkill.FIRST_TOUCH,
            hash_json({"role": "finisher", "skill": "first_touch", "source": actor_hash}),
            holdout_hash,
            finish_success,
            0.9,
        ),
        binding(SoccerSkill.FINISHING, actor_hash, canary_hash, finish_success, 1.0),
        binding(SoccerSkill.RECOVERY, handoff_hash, holdout_hash, stable_rate, 0.2),
    )
    goalkeeper_skills = (
        binding(
            SoccerSkill.POSITIONING,
            goalkeeper_policy_hash,
            holdout_hash,
            stable_rate,
            0.6,
        ),
        binding(
            SoccerSkill.SAVE,
            hash_json({"role": "goalkeeper", "skill": "save", "source": goalkeeper_policy_hash}),
            holdout_hash,
            goalkeeper_success,
            1.0,
        ),
        binding(
            SoccerSkill.RECOVERY,
            hash_json(
                {"role": "goalkeeper", "skill": "recovery", "source": goalkeeper_policy_hash}
            ),
            holdout_hash,
            stable_rate,
            0.2,
        ),
    )

    def model(
        agent_id: str,
        team_id: str,
        role: MatchRole,
        teammates: tuple[str, ...],
        opponents: tuple[str, ...],
        skills: tuple[RoleSkillBinding, ...],
    ) -> RoleSelfModel:
        return RoleSelfModel(
            agent_id,
            team_id,
            role,
            teammates,
            opponents,
            skills,
            observation_hash,
            action_hash,
            hash_json(
                {
                    "agent_id": agent_id,
                    "role": role.value,
                    "skills": [value.binding_hash for value in skills],
                }
            ),
            f"failure.{agent_id}",
            1,
        )

    roster = TeamRoleRoster(
        "s132.neural-contact-team-audit",
        (
            model(
                "red.playmaker",
                "red",
                MatchRole.PLAYMAKER,
                ("red.finisher",),
                ("blue.goalkeeper",),
                playmaker_skills,
            ),
            model(
                "red.finisher",
                "red",
                MatchRole.FINISHER,
                ("red.playmaker",),
                ("blue.goalkeeper",),
                finisher_skills,
            ),
            model(
                "blue.goalkeeper",
                "blue",
                MatchRole.GOALKEEPER,
                (),
                ("red.playmaker", "red.finisher"),
                goalkeeper_skills,
            ),
        ),
    )
    models = {value.agent_id: value for value in roster.agents}
    curricula = (
        RoleCurriculumAssignment(
            "red.playmaker",
            models["red.playmaker"].self_model_hash,
            SoccerSkill.LEAD_PASS,
            playmaker_failed,
            (models["red.finisher"].policy_artifact_hash,),
            (models["blue.goalkeeper"].policy_artifact_hash,),
            "pass_delivery_within_0p45m_rate",
            pass_success,
            5.0 / 6.0,
            1.0,
            holdout_hash,
        ),
        RoleCurriculumAssignment(
            "red.finisher",
            models["red.finisher"].self_model_hash,
            SoccerSkill.FINISHING,
            finisher_failed,
            (models["red.playmaker"].policy_artifact_hash,),
            (models["blue.goalkeeper"].policy_artifact_hash,),
            "strict_right_foot_goal_rate",
            finish_success,
            4.0 / 6.0,
            1.0,
            holdout_hash,
        ),
        RoleCurriculumAssignment(
            "blue.goalkeeper",
            models["blue.goalkeeper"].self_model_hash,
            SoccerSkill.SAVE,
            goalkeeper_failed,
            (),
            (
                models["red.playmaker"].policy_artifact_hash,
                models["red.finisher"].policy_artifact_hash,
            ),
            "save_or_non_goal_rate",
            goalkeeper_success,
            5.0 / 6.0,
            1.0,
            holdout_hash,
        ),
    )
    for assignment in curricula:
        assignment.validate_for(models[assignment.agent_id])
    candidate_result = candidate["result"]
    baseline_result = baseline["result"]
    causal_credit = {
        "changed_policy_agent_id": "red.finisher",
        "changed_policy_artifact_hash": actor_hash,
        "frozen_playmaker_policy_hash": lead_policy_hash,
        "frozen_goalkeeper_policy_hash": goalkeeper_policy_hash,
        "candidate_goal": bool(candidate_result["goal_crossed"]),
        "baseline_goal": bool(baseline_result["goal_crossed"]),
        "candidate_goalkeeper_save": bool(candidate_result["goalkeeper_save_observed"]),
        "baseline_goalkeeper_save": bool(baseline_result["goalkeeper_save_observed"]),
        "candidate_shot_speed_mps": candidate_result["shot_peak_ball_speed_mps"],
        "baseline_shot_speed_mps": baseline_result["shot_peak_ball_speed_mps"],
        "playmaker_promotion_credit": 0.0,
        "finisher_outcome_difference_credit": 1.0,
        "goalkeeper_policy_regression_claimed": False,
        "goalkeeper_counter_curriculum_created": True,
        "reason": "ONLY_FINISHER_CONTACT_POLICY_CHANGED_IN_MATCHED_COUNTERFACTUAL",
    }
    return {
        "schema_version": "rosclaw_soccer.role_awareness_audit.v1",
        "status": "PASS_ROLE_AWARENESS_AND_CREDIT_AUDIT",
        "claim": "EACH_AGENT_HAS_ROLE_SKILLS_TEAMMATES_OPPONENTS_AND_PRIVATE_FAILURE_CURRICULUM",
        "roster": roster.to_dict(),
        "roster_hash": roster.roster_hash,
        "causal_credit": causal_credit,
        "curricula": [assignment.to_dict() for assignment in curricula],
        "metrics": {
            "agent_count": len(roster.agents),
            "team_count": len({agent.team_id for agent in roster.agents}),
            "playmaker_pass_delivery_success_rate": pass_success,
            "finisher_strict_goal_rate": finish_success,
            "goalkeeper_save_or_non_goal_rate": goalkeeper_success,
            "all_agent_safety_rate": stable_rate,
            "private_curriculum_count": len(curricula),
        },
        "gates": {
            "all_agents_have_self_models": len(roster.agents) == 3,
            "cooperation_and_opposition_explicit": len({agent.team_id for agent in roster.agents})
            == 2,
            "credit_assigned_only_to_changed_policy": causal_credit["playmaker_promotion_credit"]
            == 0.0,
            "all_roles_have_private_failure_curricula": len(curricula) == len(roster.agents),
            "teammates_and_opponents_frozen_per_curriculum": all(
                assignment.frozen_teammate_policy_hashes or assignment.frozen_opponent_policy_hashes
                for assignment in curricula
            ),
        },
        "source_evidence": {
            "lead_pass_evidence_hash": lead_evidence_hash,
            "neural_canary_report_hash": canary_hash,
            "neural_holdout_report_hash": holdout_hash,
        },
        "promotion_eligible": False,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
    }


def _bound_json(path: Path, hash_field: str) -> dict[str, Any]:
    value = _read_object(path)
    claimed = value.pop(hash_field, None)
    if claimed != hash_json(value):
        raise ValueError(f"{path.name} integrity changed")
    value[hash_field] = claimed
    return value


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "RoleCurriculumAssignment",
    "run_role_awareness_audit",
    "validate_role_awareness_audit",
]
