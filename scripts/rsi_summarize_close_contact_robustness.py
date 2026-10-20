"""Require all four reviewed clusters; report paired diagnostic outcomes.

No inference, integration, optimization, promotion or media export. Conditions
are public perturbations of four consumed parent clusters, never IID/Fresh.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def contact_timing(report):
    """Observed timing, not a claim that early contact caused every later failure."""
    handoff = report["handoff_frame"]
    strike = report["strike"]
    result = {"handoff_frame": handoff}
    for kind in ("foot", "nonfoot_robot"):
        native = strike[f"first_{kind}_contact_native_index"]
        result[f"first_{kind}_contact_native_index"] = native
        result[f"{kind}_contact_before_handoff"] = (
            native is not None and handoff is not None and native < handoff * 10
        )
        result[f"{kind}_contact_without_handoff"] = native is not None and handoff is None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard", nargs=2, action="append", required=True, metavar=("PRODUCER", "REVIEW")
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-producer-head", required=True)
    parser.add_argument("--expected-review-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.output.exists() or head != args.expected_head or len(args.shard) != 4:
        raise ValueError("sealed summary source, unused output and all four shards required")
    paths = [
        p for folder in (Path.cwd() / "src", Path.cwd() / "scripts") for p in folder.rglob("*.py")
    ]
    loaded = []
    historical_pins = {}
    for producer_string, review_string in args.shard:
        producer, review = Path(producer_string), Path(review_string)
        if any(list(root.rglob("failure.json")) for root in (producer, review)):
            raise ValueError("failed artifacts cannot qualify a whole bank")
        pd = json.loads((producer / "declaration.json").read_text())
        pc = json.loads((producer / "complete.json").read_text())
        rd = json.loads((review / "declaration.json").read_text())
        rc = json.loads((review / "complete.json").read_text())
        if (
            pd["schema"] != "soccer.rsi.close_contact_perturb16.v1"
            or rd["schema"] != "soccer.rsi.close_contact_recorded_review.v1"
            or pd["source_head"] != args.expected_producer_head
            or pc["source_head"] != args.expected_producer_head
            or pc["cluster"] != pd["cluster"]
            or pc["cluster"] != rc["cluster"]
            or pc["bank_hash"] != pd["bank_hash"]
            or pc["bank_hash"] != rc["bank_hash"]
            or pc["total_new_native_steps"] != 91200
            or rc["additional_total_native_steps"] != 91200
            or pc["actual_native_steps"] != 24000
            or rc["additional_actual_native_steps"] != 24000
            or pc["forecast_native_steps"] != 67200
            or rc["additional_forecast_native_steps"] != 67200
            or len(pc["results"]) != 4
            or len(rc["results"]) != 4
            or pc["optimizer_steps"] != 0
            or rc["optimizer_steps"] != 0
            or rc["neural_policy_replayed"]
            or rc["navigation_planner_reexecuted"]
        ):
            raise ValueError("complete source-bound shard and independent review required")
        # The reader pins its own modules and the exact producer completion before imports.
        reader_sources = [
            Path(p)
            for p in rd["input_file_hashes"]
            if Path(p).name == "rsi_verify_close_contact_robustness.py"
        ]
        if len(reader_sources) != 1:
            raise ValueError("reader must bind its exact review entry point")
        reader_roots = {
            p.parent.parent
            for p in reader_sources
            if subprocess.check_output(
                ["git", "-C", str(p.parent.parent), "rev-parse", "HEAD"], text=True
            ).strip()
            == args.expected_review_head
        }
        if len(reader_roots) != 1 or rd["input_file_hashes"].get(
            str(producer / "complete.json")
        ) != digest(producer / "complete.json"):
            raise ValueError("review source and producer completion commitment differ")
        for declaration in (pd, rd):
            for p, h in declaration["input_file_hashes"].items():
                if digest(Path(p)) != h:
                    raise ValueError("bound source/raw evidence changed")
                historical_pins[p] = h
        paths += [p for root in (producer, review) for p in root.rglob("*") if p.is_file()]
        loaded.append((pc["cluster"], producer, pd, pc, rc))
    if (
        sorted(x[0] for x in loaded) != list(range(4))
        or len({x[2]["bank_hash"] for x in loaded}) != 1
    ):
        raise ValueError("exactly one shard per parent cluster and identical bank required")
    pins = {**historical_pins, **{str(p): digest(p) for p in paths}}

    from dataclasses import asdict

    from rosclaw_soccer.training.close_contact_courses import close_contact_perturbations
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_atomic_artifacts import write_once

    bank = close_contact_perturbations()
    bank_hash = preflight_receiving_courses(bank)
    if bank_hash != loaded[0][2]["bank_hash"]:
        raise ValueError("aggregate whole-bank identity differs")
    clusters = []
    for cluster, _producer, _pd, pc, rc in sorted(loaded):
        rows = []
        for item, pair, reviewed in zip(
            [x for x in bank if x.parent_index == cluster],
            pc["results"],
            rc["results"],
            strict=True,
        ):
            course = asdict(item) | {"course": list(item.course)}
            if pair["course"] != course or reviewed["course"] != course:
                raise ValueError("every declared variant must appear exactly once")
            for actual, proof in zip(pair["results"], reviewed["results"], strict=True):
                if (
                    actual["arm"] != proof["arm"]
                    or actual["score"] != proof["score"]
                    or actual["selected_index"] != proof["selected_index"]
                    or not proof["actual_physics_bit_exact"]
                    or not proof["all_candidate_physics_bit_exact"]
                    or not proof["posthandoff_contact_reference_reconstructed"]
                    or proof["candidate_count"] != (6 if actual["arm"] == "full_teacher" else 0)
                ):
                    raise ValueError("whole physical/reference review differs")
            nominal, teacher = pair["results"]
            flags = dict(
                rescue=not nominal["score"]["qualified"] and teacher["score"]["qualified"],
                regression=nominal["score"]["qualified"] and not teacher["score"]["qualified"],
                added_unsafe=nominal["score"]["physical_prefix_and_suffix_safe"]
                and not teacher["score"]["physical_prefix_and_suffix_safe"],
            )
            if any(pair[k] != value or reviewed[k] != value for k, value in flags.items()):
                raise ValueError("paired outcome flags differ")
            rows.append(
                dict(
                    course=course,
                    **flags,
                    nominal=dict(score=nominal["score"], **contact_timing(nominal)),
                    teacher=dict(
                        score=teacher["score"],
                        selected_name=teacher["selected_name"],
                        uncertified_nominal_fallback=teacher["uncertified_nominal_fallback"],
                        **contact_timing(teacher),
                    ),
                )
            )
        clusters.append(
            dict(
                parent_cluster=cluster,
                conditions=rows,
                nominal_qualified=sum(r["nominal"]["score"]["qualified"] for r in rows),
                teacher_qualified=sum(r["teacher"]["score"]["qualified"] for r in rows),
                rescued=sum(r["rescue"] for r in rows),
                regressed=sum(r["regression"] for r in rows),
                added_unsafe=sum(r["added_unsafe"] for r in rows),
            )
        )
    for p, h in pins.items():
        if digest(Path(p)) != h:
            raise ValueError("sealed summary inputs changed")
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            schema="soccer.rsi.close_contact_reviewed_summary.v1",
            source_head=head,
            input_file_hashes=pins,
            bank_hash=bank_hash,
            independent_parent_clusters=4,
            declared_public_conditions=16,
            private_fresh_accessed=False,
            inferential_iid_claim=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    report = dict(
        clusters=clusters,
        nominal_qualified=sum(x["nominal_qualified"] for x in clusters),
        teacher_qualified=sum(x["teacher_qualified"] for x in clusters),
        rescued=sum(x["rescued"] for x in clusters),
        regressed=sum(x["regressed"] for x in clusters),
        added_unsafe=sum(x["added_unsafe"] for x in clusters),
        conditions=16,
        parent_clusters=4,
        producer_actual_native_steps=96000,
        producer_forecast_native_steps=268800,
        independent_recorded_replay_native_steps=364800,
        total_charged_native_steps=729600,
        new_native_steps_in_this_summary=0,
        optimizer_steps=0,
        official_exam_qualified=False,
        learned_student_qualified=False,
        continuous_team_qualified=False,
        publicity_license_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.output / "complete.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "clusters"}), flush=True)


if __name__ == "__main__":
    main()
