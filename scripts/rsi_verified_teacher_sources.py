"""Join independently verified nominal/perturbed SIM evidence without relabeling."""

import json


def admit_teacher_sources(specs):
    from scripts.rsi_native_context_recovery_common import digest

    if [s[0] for s in specs] != ["nominal", "perturbed"]:
        raise ValueError("one nominal and one separate perturbation study required")
    pins, sources = {}, []
    for kind, audit_path, expected, root in specs:
        if digest(audit_path) != expected:
            raise ValueError("exact independently closed teacher audit required")
        audit = json.loads(audit_path.read_text())
        if (
            audit["status"] != "MOTION_SYNERGY_TEACHER_RAW_CEM_FUTURES_AND_FULL_EXAM_VERIFIED"
            or audit["actual_worlds"] != 80
            or not audit["expanded_public_validation"]
            or audit.get("perturbed_public_validation", False) != (kind == "perturbed")
            or (kind == "nominal" and not audit["qualified_on_expanded_public"])
        ):
            raise ValueError(
                "complete nominal qualification and separate perturbation audit required"
            )
        for key, value in audit["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting nominal/perturbed provenance")
            pins[key] = value
        pins[str(audit_path)] = expected
        cp = root / "complete.json"
        if pins.get(str(cp)) != digest(cp):
            raise ValueError("exact physical producer join required")
        producer = json.loads(cp.read_text())
        if [r["course_index"] for r in producer["results"]] != list(range(40)):
            raise ValueError("all40 contexts required, no good-case filtering")
        sources.append((kind, root, producer, audit))
    return pins, sources
