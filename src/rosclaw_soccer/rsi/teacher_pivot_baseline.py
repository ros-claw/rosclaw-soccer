"""Read-only admission of an audited limited teacher, before a new SIM probe."""

import hashlib
import json
from pathlib import Path


def audited_teacher_baseline(
    root: Path, *, head: str, training_hash: str
) -> tuple[str, dict[str, str]]:
    def digest(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    path = root / "complete.json"
    if path.is_symlink() or (root / "failed.json").exists():
        raise ValueError("complete ordinary teacher baseline audit required")
    a = json.loads(path.read_text())
    if (
        a["status"] != "DUAL_MOTOR_ASSISTED18_CONTINUAL_OWN_NATIVE_AUDIT_PASS"
        or a["source_head"] != head
        or a["worlds"] != 18
        or len(a["results"]) != 18
        or a["teacher_mode"] != "measured_contact_observed_shin"
        or not 0 <= a["selected_teacher_successes"] < 3
        or a["hardware_authorized"] is not False
        or a["promotion_authorized"] is not False
    ):
        raise ValueError("actual audited incomplete three-context teacher required")
    pins = dict(a["input_file_hashes"])
    sources = [
        Path(p) for p, h in pins.items() if h == a["source_hash"] and p.endswith("/complete.json")
    ]
    if len(sources) != 1 or sources[0].is_symlink() or digest(sources[0]) != a["source_hash"]:
        raise ValueError("unique exact physical teacher source required")
    r = json.loads(sources[0].read_text())
    if (
        r["status"] != "DUAL_MOTOR_ACTUAL_STATE_TEACHER_RECOVERY_CLOSED_PENDING_OWN_AUDIT"
        or r["hardware_authorized"] is not False
        or r["promotion_authorized"] is not False
        or (sources[0].parent / "failed.json").exists()
        or r["source_head"] != head
        or r["training_hash"] != training_hash
        or a["counts"]["native"] != r["actual_native_steps"]
        or a["counts"]["neural"] != r["actual_actor_decisions"]
        or a["selected_teacher_successes"] != r["selected_teacher_successes"]
    ):
        raise ValueError("same actually examined learner and teacher work required")
    h = digest(path)
    pins[str(path)] = h
    pins[str(sources[0])] = a["source_hash"]
    return h, pins
