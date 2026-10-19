"""Read-only declared pipeline roles, not scheduler or proof of science."""

from typing import Any

MILESTONES = {"terminal_complete", "factory-contract.json"}


def annotate_dependencies(
    campaigns: dict[str, dict[str, Any]], plan: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if (
        type(plan) is not dict
        or plan.get("schema") != "soccer.research.observed_pipeline_dependencies.v1"
        or type(plan.get("dependencies")) is not dict
        or set(plan["dependencies"]) != set(campaigns)
        or not 1 <= len(campaigns) <= 128
    ):
        raise ValueError("bounded exact observed campaign dependency map required")
    dependencies = plan["dependencies"]
    for unit, requirements in dependencies.items():
        if type(requirements) is not list or len(requirements) > 16:
            raise ValueError("bounded explicit prerequisite list required")
        seen = set()
        for requirement in requirements:
            if (
                type(requirement) is not dict
                or set(requirement) != {"unit", "milestone"}
                or type(requirement["unit"]) is not str
                or requirement["unit"] not in campaigns
                or requirement["unit"] == unit
                or type(requirement["milestone"]) is not str
                or requirement["milestone"] not in MILESTONES
            ):
                raise ValueError("known distinct predecessor and allowlisted milestone required")
            key = (requirement["unit"], requirement["milestone"])
            if key in seen:
                raise ValueError("duplicate prerequisite")
            seen.add(key)
    active: set[str] = set()
    visited: set[str] = set()

    def visit(unit: str) -> None:
        if unit in active:
            raise ValueError("acyclic scientific pipeline required")
        if unit in visited:
            return
        active.add(unit)
        for requirement in dependencies[unit]:
            visit(requirement["unit"])
        active.remove(unit)
        visited.add(unit)

    for unit in campaigns:
        visit(unit)
    result = {}
    for unit, observation in campaigns.items():
        pending = []
        failed = []
        for requirement in dependencies[unit]:
            predecessor = campaigns[requirement["unit"]]
            service = predecessor.get("service", {})
            artifacts = predecessor.get("artifacts") or {}
            is_failed = (
                predecessor.get("phase") in ("FAILED", "OBSERVATION_FAILED")
                or service.get("ActiveState") == "failed"
                or artifacts.get("failure_present")
            )
            if is_failed:
                failed.append(requirement)
                continue
            if requirement["milestone"] == "terminal_complete":
                done = (
                    service.get("LoadState") == "loaded"
                    and service.get("MainPID") == "0"
                    and service.get("ActiveState") in ("active", "inactive")
                    and service.get("Result") == "success"
                    and artifacts.get("complete_present") is True
                )
            else:
                done = requirement["milestone"] in artifacts.get("checkpoint_names", [])
            if not done:
                pending.append(requirement)
        if observation.get("phase") in ("FAILED", "OBSERVATION_FAILED"):
            role = "OWN_FAILURE_NEEDS_REVIEW"
        elif failed:
            role = "DEPENDENCY_FAILED_NEEDS_AGENT_REVIEW"
        elif pending:
            role = "WAITING_FOR_DECLARED_PREDECESSOR"
        elif (observation.get("artifacts") or {}).get("complete_present"):
            role = "COMPLETE_METADATA_REQUIRES_AUTHENTICATION"
        else:
            role = "PREREQUISITES_OBSERVED_STAGE_EXECUTION_NOT_VERIFIED"
        result[unit] = dict(
            **observation,
            declared_work_role=role,
            pending_prerequisites=pending,
            failed_prerequisites=failed,
            dependency_contents_verified=False,
            goal_complete=False,
        )
    return result
