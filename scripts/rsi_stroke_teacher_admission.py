"""Read-only binding to a complete independently replayed stroke teacher."""

import json
from pathlib import Path

from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_stroke_search_common import admit


def admit_teacher(search_root, audit_root, audit_hash):
    sp, ap, dp = (
        search_root / "complete.json",
        audit_root / "complete.json",
        search_root / "declaration.json",
    )
    if any(p.is_symlink() for p in (sp, ap, dp)) or digest(ap) != audit_hash:
        raise ValueError("exact independently audited coherent teacher required")
    s, a, d = [json.loads(p.read_text()) for p in (sp, ap, dp)]
    if (
        a["status"] != "SHARED_STROKE72_ALL_OWN_NATIVE_AND_CEM_SELECTION_AUDIT_PASS"
        or a["source_hash"] != digest(sp)
        or s["declaration_hash"] != digest(dp)
        or a["worlds"] != 72
        or a["counts"]["native"] != s["actual_native_steps"]
        or a["counts"]["neural"] != s["actual_neural_decisions"]
        or not a["independent_whole_world_replay"]
        or a["selected_teacher_successes"] != 2
        or s["selected_teacher_successes"] != 2
        or a["unsearched_control_quality"] is not True
        or s["unsearched_control_quality"] is not True
        or a["training_hash"] != s["training_hash"]
        or any(
            r["student_success_claim"] is not False
            or r["PPO_actor_rows"] != 0
            or r["promotion_authorized"] is not False
            or r["hardware_authorized"] is not False
            for r in (s, a)
        )
    ):
        raise ValueError(
            "fully replayed two-context and additional-context teacher competence required"
        )
    from rosclaw_soccer.rsi.stroke_search import verify_search_selection

    verify_search_selection(s)
    pins = dict(a["input_file_hashes"])

    def receipt(h):
        ps = [Path(p) for p, v in pins.items() if v == h and p.endswith("/complete.json")]
        if len(ps) != 1:
            raise ValueError("unique original source binding required")
        return ps[0]

    ep, op = receipt(d["source_exam_hash"]), receipt(d["source_audit_hash"])
    original, training_hash, contexts, centers, _ = admit(
        ep.parent, op.parent, d["source_audit_hash"]
    )
    if training_hash != s["training_hash"]:
        raise ValueError("same unmodified learner motor required")
    for p, h in original.items():
        if p in pins and pins[p] != h:
            raise ValueError("conflicting motor evidence")
        pins[p] = h
    for p in (sp, ap, dp):
        pins[str(p)] = digest(p)
    return pins, training_hash, contexts, centers, s["selected_latent"]
