"""Read-only prerequisite for a separately declared, still unqualified SIM domain."""

import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.providers.g1.velocity_policy import file_hash as digest


def admit_heading_evidence(path: Path, expected_hash: str) -> tuple[dict[str, Any], dict[str, str]]:
    if path.is_symlink() or path.stat().st_size > 262144 or digest(path) != expected_hash:
        raise ValueError("exact bounded experimental evidence declaration required")
    cfg = json.loads(path.read_text())
    if set(cfg) != {"schema", "point_root", "point_hash", "reference_source", "reference_hash"} or (
        cfg["schema"] != "soccer.native_single_finisher_experimental_approach.v1"
    ):
        raise ValueError("explicit existing heading-held approach evidence required")
    root = Path(cfg["point_root"])
    rp, dp = root / "complete.json", root / "declaration.json"
    if rp.is_symlink() or dp.is_symlink() or digest(rp) != cfg["point_hash"]:
        raise ValueError("exact completed experimental point evidence required")
    report, declaration = json.loads(rp.read_text()), json.loads(dp.read_text())
    expected_rows = [
        (course, kind, repeat)
        for course in (
            "standing",
            "forward-1",
            "forward-1p5",
            "forward-2",
            "lateral-positive-0p8",
            "lateral-negative-0p8",
        )
        for kind in (
            ("legacy", "experimental") if course in ("standing", "forward-1") else ("experimental",)
        )
        for repeat in range(2)
    ]
    if (
        report["status"] != "EXPERIMENTAL_VELOCITY_POINTS_MEASURED_NOT_A_DOMAIN_QUALIFICATION"
        or declaration["protocol"] != "native_one_body_experimental_velocity_heading_hold.v2"
        or len(report["records"]) != 16
        or [(r["course"], r["kind"], r["repeat"]) for r in report["records"]] != expected_rows
        or not all(
            all(
                r[k] is True
                for k in (
                    "point_feasible",
                    "no_ball_contact",
                    "healthy_full_horizon",
                    "actual_effort_bound_pass",
                )
            )
            and r["native_steps"] == 6000
            for r in report["records"]
        )
        or not all(
            all(r["fresh_reexecution_exact"].values()) for r in report["records"] if r["repeat"]
        )
    ):
        raise ValueError("all declared heading-held points and reexecutions must have passed")
    pins = dict(report["input_file_hashes"])
    pins.update({str(path): expected_hash, str(rp): cfg["point_hash"], str(dp): digest(dp)})
    ref = Path(cfg["reference_source"])
    if ref.is_symlink() or digest(ref) != cfg["reference_hash"]:
        raise ValueError("exact upstream training configuration required")
    pins[str(ref)] = cfg["reference_hash"]
    current_codec = Path(__file__).parents[1] / "providers/g1/experimental_velocity_envelope.py"
    if {
        v
        for k, v in report["input_file_hashes"].items()
        if k.endswith("/providers/g1/experimental_velocity_envelope.py")
    } != {digest(current_codec)}:
        raise ValueError("current codec differs from physically measured point codec")
    for row in report["records"]:
        artifact = root / row["artifact"]["path"]
        if (
            artifact.parent.resolve() != root.resolve()
            or artifact.is_symlink()
            or (digest(artifact) != row["artifact"]["hash"])
        ):
            raise ValueError("exact confined point trajectory required")
        pins[str(artifact)] = row["artifact"]["hash"]
    for name, expected in pins.items():
        if digest(Path(name)) != expected:
            raise ValueError("point evidence ancestry changed: " + name)
    return dict(
        experimental_fast=True, reference_source=ref, reference_hash=cfg["reference_hash"]
    ), pins
