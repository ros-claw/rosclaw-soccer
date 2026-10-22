"""Bind all three audited students and the already qualified physical teacher."""

import json
from pathlib import Path

from scripts.rsi_native_context_recovery_common import digest


def admit_prefix_study(data_root, exam_roots):
    from rosclaw_soccer.rsi.contact_consolidation_data import BANK_STATUS
    from scripts.rsi_stroke_teacher_admission import admit_teacher

    bp = data_root / "complete.json"
    bank = json.loads(bp.read_text())
    if bp.is_symlink() or bank["status"] != BANK_STATUS or len(bank["records"]) != 65:
        raise ValueError("complete exact contact bank65 required")
    pins = dict(bank["input_file_hashes"])
    pins[str(bp)] = digest(bp)

    def receipt(h):
        paths = [Path(p) for p, v in pins.items() if v == h and p.endswith("/complete.json")]
        if len(paths) != 1 or digest(paths[0]) != h:
            raise ValueError("unique exact prefix study lineage required")
        return paths[0], json.loads(paths[0].read_text())

    ap, audit = receipt(bank["robustness_audit_hash"])
    if (
        audit["status"] != "FIXED_STROKE_PAIRED48_ALL96_OWN_NATIVE_AUDIT_PASS"
        or audit["summary"]["teacher_locally_qualified"] is not True
    ):
        raise ValueError("independently qualified robust teacher required")
    pp, _ = receipt(audit["source_hash"])
    robust_decl = json.loads((pp.parent / "declaration.json").read_text())
    sp, _ = receipt(robust_decl["search_receipt_hash"])
    sap, _ = receipt(robust_decl["search_audit_hash"])
    tpins, warm_hash, _, centers, latent = admit_teacher(
        sp.parent, sap.parent, robust_decl["search_audit_hash"]
    )
    for p, h in tpins.items():
        if p in pins and pins[p] != h:
            raise ValueError("conflicting original teacher pin")
        pins[p] = h
    if bank["warm_start_hash"] != warm_hash or len(exam_roots) != 3:
        raise ValueError("same W6 and all three independent student seeds required")
    students, courses = [], []
    for expected_seed, root in zip((65023, 65024, 65025), exam_roots, strict=True):
        cp, np, op, dp = (
            root / "complete.json",
            root / "native-exam/complete.json",
            root / "native-audit/complete.json",
            root / "native-exam/declaration.json",
        )
        if any(p.is_symlink() for p in (cp, np, op, dp)) or any(
            (r / "failed.json").exists() for r in (root, np.parent, op.parent)
        ):
            raise ValueError("complete unfailing independently audited source examination required")
        chain, native, own, declaration = [json.loads(p.read_text()) for p in (cp, np, op, dp)]
        if (
            chain["status"] != "CONTACT_CONSOLIDATION_EXAM_CHAIN_CLOSED_NOT_GOAL_COMPLETE"
            or own["status"] != "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS"
            or own["source_hash"] != digest(np)
            or native["declaration_hash"] != digest(dp)
            or own["counts"]["native"] != native["actual_native_steps"]
            or own["counts"]["neural"] != native["actual_actor_decisions"]
            or own["worlds"] != 44
            or len(own["results"]) != 44
            or own["checkpoint_kind"] != "contact-consolidated"
            or own["training_hash"] != chain["training_hash"]
            or declaration["checkpoint_kind"] != "contact-consolidated"
            or declaration["online_teacher_enabled"] is not False
            or any(
                v[k] is not False
                for v in (chain, native, own, declaration)
                for k in ("hardware_authorized", "promotion_authorized")
            )
        ):
            raise ValueError("all44 whole own-native student histories required")
        for path, h in own["input_file_hashes"].items():
            if path in pins and pins[path] != h:
                raise ValueError("conflicting student lineage")
            pins[path] = h
        for path in (cp, np, op, dp):
            pins[str(path)] = digest(path)
        training_path, training = receipt(chain["training_hash"])
        if (
            training["config"]["seed"] != expected_seed
            or training["warm_start_hash"] != warm_hash
            or training["data_hash"] != digest(bp)
        ):
            raise ValueError("all preregistered independently sampled same-data students required")
        students.append(
            dict(
                seed=expected_seed,
                training_root=str(training_path.parent),
                training_hash=chain["training_hash"],
                exam_root=str(root),
            )
        )
        for i in (4, 5, 8, 17, 20, 21):
            ctx = dict(declaration["contexts"][i], context_id=i)
            row = next(x for x in native["records"] if x["context"] == i and x["repeat"] == 0)
            if ctx["case"]["bank"] != "train" or ctx["case"]["not_fresh"] is not True:
                raise ValueError("consumed TRAIN sources only, never Fresh")
            for art in row["artifacts"].values():
                path = np.parent / art["path"]
                if (
                    path.is_symlink()
                    or not path.resolve().is_relative_to(np.parent.resolve())
                    or digest(path) != art["hash"]
                ):
                    raise ValueError("confined exact actual student trajectory required")
                pins[str(path)] = art["hash"]
            for prefix in (40, 80):
                courses.append(
                    dict(
                        index=len(courses),
                        seed=expected_seed,
                        context=i,
                        ctx=ctx,
                        prefix_controls=prefix,
                        baseline_quality=row["healthy_quality"],
                        baseline_kick_path=str(np.parent / row["artifacts"]["kick"]["path"]),
                        baseline_kick_hash=row["artifacts"]["kick"]["hash"],
                        center=centers.get(i, centers[4]),
                        latent=latent if i in (4, 20, 21) else [[0.0] * 6, [0.0] * 6],
                    )
                )
    return pins, warm_hash, students, courses
