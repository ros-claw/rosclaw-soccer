"""Authenticate closed motor-memory experiments before perturbation allocation."""

import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.paid_canonical_behavior import DATA_STATUS, SCHEMA, digest, read_witnesses
from rosclaw_soccer.rsi.paid_neighborhood_study import (
    contact_neighborhood_cases,
    neighborhood_config,
)


def load_neighborhood_sources(
    data_root: Path, data_hash: str, pipeline_root: Path, pipeline_hash: str
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    pins: dict[str, str] = {}

    def read(root: Path, h: str) -> dict[str, Any]:
        path = root / "complete.json"
        if path.is_symlink() or digest(path) != h or (root / "failed.json").exists():
            raise ValueError("exact closed perturbation source required")
        r: dict[str, Any] = json.loads(path.read_text())
        if any(r.get(k) is not False for k in ("hardware_authorized", "promotion_authorized")):
            raise ValueError("SIM-only perturbation source required")
        for f, fh in r.get("input_file_hashes", {}).items():
            if f in pins and pins[f] != fh:
                raise ValueError("conflicting perturbation lineage")
            pins[f] = fh
        pins[str(path)] = h
        return r

    data = read(data_root, data_hash)
    queue = read(pipeline_root, pipeline_hash)
    if (
        data.get("status") != DATA_STATUS
        or data.get("schema") != SCHEMA
        or queue.get("status") != "PAID_CANONICAL_MEMORY_ALL_DECLARED_LANES_CLOSED_NOT_PROMOTED"
        or queue.get("data_hash") != data_hash
        or data.get("bank") != "train"
        or data.get("Fresh_fit_rows") != 0
        or data.get("DEV_fit_rows") != 0
    ):
        raise ValueError("closed canonical reference study required, not a queued fit")
    fits = {}
    for model in ("learned", "zero-control"):
        ref = queue["fits"][model]
        root = pipeline_root / (model + "-trained")
        if ref["root"] != str(root):
            raise ValueError("confined original fitting identity required")
        r = read(root, ref["hash"])
        if (
            r["data_hash"] != data_hash
            or r["config"].get("canonical_behavior") is not True
            or r["config"].get("control") is not (model == "zero-control")
            or r["body_scene_hash"] != data["body_scene_hash"]
            or r["optimizer_updates"] != 0
        ):
            raise ValueError("same-data learned/control memory pair required")
        fits[model] = dict(root=str(root), hash=ref["hash"])
        cp = Path(r["checkpoint_path"])
        if cp.is_symlink() or cp.parent != root or digest(cp) != r["checkpoint_hash"]:
            raise ValueError("confined exact memory checkpoint required")
        pins[str(cp)] = r["checkpoint_hash"]
    train_plan = None
    for name in ("learned-train20", "learned-train16", "learned-dev16", "zero-control-train20"):
        refs = queue["lanes"][name]
        objects = []
        for suffix in ("exam", "audit"):
            root = pipeline_root / (name + "-" + suffix)
            if refs[suffix]["root"] != str(root):
                raise ValueError("confined physical evidence required")
            objects.append(read(root, refs[suffix]["hash"]))
        exam, audit = objects
        model = "zero-control" if name.startswith("zero-control") else "learned"
        size = 20 if name.endswith("train20") else 16
        prefix = "PAID_ROUTER_TRAIN20" if size == 20 else "PAID_FULL_ENTRY_ORIGINAL16"
        if (
            exam["status"] != prefix + "_WORLDS_CLOSED_PENDING_AUDIT"
            or audit["status"] != prefix + "_OWN_NN_AND_ALL_PHASE_PHYSICS_AUDIT_PASS"
            or exam["student_hash"] != fits[model]["hash"]
            or audit["student_hash"] != exam["student_hash"]
            or audit["source_hash"] != refs["exam"]["hash"]
            or audit["summary"] != exam["summary"]
            or audit["counts"]["worlds"] != 2 * size
            or audit["own_native_steps"] != exam["actual_native_steps"]
            or audit["own_trajectory_neural_rows"] != exam["actual_neural_rows"]
            or [w["case"] for w in exam["worlds"]] != list(range(size))
            or exam["summary"]["stable"] != size
        ):
            raise ValueError("complete ownNN and all-phase native audit required")
        dp = pipeline_root / (name + "-exam") / "declaration.json"
        if dp.is_symlink() or digest(dp) != exam["declaration_hash"]:
            raise ValueError("bound first-enabled declaration required")
        d = json.loads(dp.read_text())
        if d.get("first_enabled") is not True or d["schedule"] != ["student"] * 3:
            raise ValueError("actual first-enabled student required")
        pins[str(dp)] = digest(dp)
        if name == "learned-train20":
            if any(not exam["worlds"][i]["quality_by_kick"][0] for i in (4, 5)):
                raise ValueError("both original hard first contacts must pass before expansion")
            pr = read(Path(d["plan_root"]), d["plan_hash"])
            pdp = Path(d["plan_root"]) / "declaration.json"
            if pdp.is_symlink() or digest(pdp) != pr["declaration_hash"]:
                raise ValueError("bound original source geometries required")
            pins[str(pdp)] = digest(pdp)
            train_plan = json.loads(pdp.read_text())
    if train_plan is None:
        raise ValueError("complete source bank required")
    identity = {
        k: data[k]
        for n in ("original", "teacher", "teacher_audit", "replan", "replan_audit")
        for k in (n + "_root", n + "_hash")
    }
    sources, source_pins = read_witnesses(identity)
    for f, h in source_pins.items():
        if f in pins and pins[f] != h:
            raise ValueError("conflicting audited teacher identity")
        pins[f] = h
    rd = json.loads((Path(data["replan_root"]) / "declaration.json").read_text())
    teacher_actor_root = Path(rd["source_arguments"]["student_root"])
    teacher_actor_hash = rd["source_arguments"]["student_hash"]
    teacher_actor = read(teacher_actor_root, teacher_actor_hash)
    cp = Path(teacher_actor["checkpoint_path"])
    if (
        cp.is_symlink()
        or cp.parent != teacher_actor_root
        or digest(cp) != teacher_actor["checkpoint_hash"]
    ):
        raise ValueError("bound old student for the successful teacher prefix required")
    pins[str(cp)] = teacher_actor["checkpoint_hash"]
    specifications: dict[str, Any] = dict(
        models={
            "frozen_parent": fits["zero-control"],
            "learned_memory": fits["learned"],
            "fixed_successful_teacher": dict(root=str(teacher_actor_root), hash=teacher_actor_hash),
        },
        teachers={},
    )
    for c in data["choices"]:
        selected = c["selected"]
        case = selected["case"]
        fixed = sources["teacher"]["searches"][0]
        specifications["teachers"][str(case)] = dict(
            center=fixed["center"],
            switch_control=0 if selected["kind"] == "original" else selected["switch_control"],
            latent=fixed["latent"]
            if selected["kind"] == "original"
            else selected["record"]["latent"],
        )
    cases = contact_neighborhood_cases([train_plan["cases"][i] for i in (4, 5)])
    return data, cases, pins, specifications


def load_neighborhood_plan(
    root: Path, expected: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        rp.is_symlink()
        or dp.is_symlink()
        or digest(rp) != expected
        or (root / "failed.json").exists()
    ):
        raise ValueError("exact closed neighborhood preflight required")
    r, d = [json.loads(p.read_text()) for p in (rp, dp)]
    if (
        r["status"] != "PAID_CONTACT_NEIGHBORHOOD32_PREREGISTERED_NO_ROLLOUT"
        or r["declaration_hash"] != digest(dp)
        or r["cases_preflighted"] != 32
        or r["own_native_steps"] != 0
        or r["own_neural_rows"] != 0
        or r["optimizer_updates"] != 0
        or r["config"] != neighborhood_config()
        or any(r[k] != v for k, v in d.items())
        or any(
            v.get(k) is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("complete frozen perturbation design required")
    data, cases, source_pins, specs = load_neighborhood_sources(
        Path(d["data_root"]), d["data_hash"], Path(d["pipeline_root"]), d["pipeline_hash"]
    )
    if (
        d["cases"] != cases
        or d["specifications"] != specs
        or d["body_scene_hash"] != data["body_scene_hash"]
    ):
        raise ValueError("reconstructed perturbation declaration differs")
    pins = dict(r["input_file_hashes"])
    for f, h in source_pins.items():
        if pins.get(f) != h:
            raise ValueError("bound original perturbation sources differ")
    if len(r["entries"]) != 32 or [v["case"] for v in r["entries"]] != list(range(32)):
        raise ValueError("every declared physical entry required")
    for link in r["entries"]:
        path = root / f"case{link['case']}-initial.npz"
        if link["raw_path"] != path.name or path.is_symlink() or digest(path) != link["raw_hash"]:
            raise ValueError("exact preflighted initial state required")
        pins[str(path)] = link["raw_hash"]
    pins[str(rp)], pins[str(dp)] = expected, digest(dp)
    return r, d, pins
