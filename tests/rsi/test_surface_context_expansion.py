import hashlib
import json
from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.public_physics_option_planning import planning_contexts
from rosclaw_soccer.rsi.surface_context_expansion import (
    context_jobs,
    require_context_baselines,
    require_previous_surface,
)
from scripts.rsi_read_mesh_contact_native_pilot import bounded_profile


def test_fixed_six_contexts_match_independent_reader_declaration():
    contexts = planning_contexts(
        PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), "sha256:" + "a" * 64, 18)
    )
    jobs = context_jobs(contexts)
    assert jobs == bounded_profile("surface-context-six")["jobs"]
    assert [j["context_index"] for j in jobs] == list(range(2, 8))
    assert [j["group"] for j in jobs] == [68, 69, 72, 73, 76, 77]
    with pytest.raises(ValueError):
        context_jobs(contexts[:6])
    with pytest.raises(ValueError):
        context_jobs(contexts[::-1])
    assert all(j["role"] == "teacher" for j in jobs)


@pytest.mark.parametrize(
    "tamper", [None, "config", "complete", "reader", "parity", "control", "world", "fit"]
)
def test_real_byte_predecessor_and_default_off_qualification_required(tmp_path, tamper):
    old, current, output = (tmp_path / n for n in ("old", "current", "output"))
    for p in (old, current, output):
        p.mkdir()

    def digest(path):
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    paths = (
        "rsi/native_first_touch_episode.py",
        "rsi/first_touch_campaign_worker.py",
        "rsi/cpu_motor_transfer_evidence.py",
        "rsi/mesh_surface_swing_probe.py",
        "rsi/mesh_surface_swing_evidence.py",
        "rsi/mesh_surface_swing.py",
        "physics/convex_surface_distance.py",
    )
    for relative in paths:
        for root in (old, current):
            path = root / "src/rosclaw_soccer" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic control source")
    old_cfg = dict(
        output_root=str(output),
        source_root=str(old),
        mesh_task_contact_world_hash="world",
        portfolio_fit_hash="fit",
    )
    path = old / "config.json"
    path.write_text(json.dumps(old_cfg))
    reader = old / "reader.py"
    reader.write_text("synthetic terminal reader")
    documents = {
        output / "complete.json": dict(
            report_hash="complete", declaration_hash="declaration", results=[{}] * 7
        ),
        output / "declaration.json": dict(old_cfg, report_hash="declaration"),
        old / "audit.json": dict(
            report_hash="audit",
            complete_hash="complete",
            schema="soccer.rsi.mesh_surface_swing_terminal_readback.v1",
            reader_source_hash=digest(reader),
            source_native_bodies=7,
            whole_input_and_witness_files_verified=True,
            shared_payload_logical_seals_rebuilt=True,
            prior_independent_replays_authenticated=True,
            all_original_pre_activation_prefixes_bit_equal=True,
            disabled_complete_300_frame_noninterference_verified=True,
        ),
    }
    cfg = dict(
        mesh_task_contact_world_hash="world",
        portfolio_fit_hash="fit",
        surface_predecessor=dict(
            configuration=str(path),
            configuration_file_hash=digest(path),
            complete_hash="complete",
            readback=str(old / "audit.json"),
            readback_hash="audit",
            reader_source=str(reader),
        ),
    )
    audit = documents[old / "audit.json"]
    if tamper == "config":
        cfg["surface_predecessor"]["configuration_file_hash"] = "different"
    elif tamper == "complete":
        documents[output / "complete.json"]["results"].pop()
    elif tamper == "reader":
        reader.write_text("changed reader")
    elif tamper == "parity":
        audit["disabled_complete_300_frame_noninterference_verified"] = False
    elif tamper == "control":
        (current / "src/rosclaw_soccer" / paths[0]).write_text("changed physical control")
    elif tamper == "world":
        cfg["mesh_task_contact_world_hash"] = "other world"
    elif tamper == "fit":
        cfg["portfolio_fit_hash"] = "other fit"
    for document_path, contents in documents.items():
        document_path.write_text(json.dumps(contents))
    if tamper:
        with pytest.raises(ValueError):
            require_previous_surface(cfg, current, documents.__getitem__, digest, {})
    else:
        pins = {}
        assert (
            require_previous_surface(cfg, current, documents.__getitem__, digest, pins) == "audit"
        )
        assert len(pins) == 19


def test_all_context_objects_required_before_new_screen():
    with pytest.raises(ValueError):
        context_jobs([SimpleNamespace(context_index=0)] * 8)


@pytest.mark.parametrize(
    "tamper", [None, "count", "audit", "membership", "identity", "hash", "reader", "partial"]
)
def test_all_twelve_original_endpoints_authenticated_before_new_physics(tmp_path, tamper):
    def digest(path):
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    jobs = bounded_profile("surface-context-six")["jobs"]
    results = [{}, {}, {}, {}]
    items, pins, documents = [], {}, {}
    for j in jobs:
        for role in ("control", "teacher"):
            prior = dict(
                group=j["group"],
                job=role + "-" + j["pair_id"],
                receipt_hash=role + str(j["group"]),
                outcome=dict(task_pass=False),
            )
            results.append(prior)
            if role == "teacher":
                p = tmp_path / (prior["job"] + ".json")
                p.write_text(json.dumps(prior))
                documents[p] = prior
                items.append(
                    dict(
                        path=str(p),
                        receipt_hash=prior["receipt_hash"],
                        file_hash=digest(p),
                        pair_id=j["pair_id"],
                    )
                )
    results += [{}, {}]
    reader = tmp_path / "reader.py"
    reader.write_text("original reader")
    complete = dict(report_hash="complete", results=results)
    audit = dict(
        report_hash="audit",
        complete_hash="complete",
        schema="soccer.rsi.mesh_option4_teacher_screen_terminal_readback.v1",
        reader_source_hash=digest(reader),
        source_native_bodies=18,
        whole_input_and_witness_files_verified=True,
        shared_payload_logical_seals_rebuilt=True,
        prior_independent_replays_authenticated=True,
        rows=list(results[4:16]),
    )
    documents[tmp_path / "complete.json"] = complete
    documents[tmp_path / "audit.json"] = audit
    cfg = dict(
        baseline_receipts=items,
        baseline_terminal_readback=str(tmp_path / "audit.json"),
        baseline_terminal_readback_hash="audit",
        baseline_reader_source=str(reader),
    )
    if tamper == "count":
        complete["results"].pop()
    elif tamper == "audit":
        audit["whole_input_and_witness_files_verified"] = False
    elif tamper == "membership":
        audit["rows"] = audit["rows"][1:]
    elif tamper == "identity":
        items[0]["receipt_hash"] = "other"
    elif tamper == "hash":
        items[0]["file_hash"] = "other"
    elif tamper == "reader":
        reader.write_text("changed reader")
    elif tamper == "partial":
        jobs = jobs[:-1]
    for p, d in documents.items():
        p.write_text(json.dumps(d))

    # Seals are independently tested by the caller; these tests exercise the
    # whole audited membership and exact endpoint selection before allocation.
    def sealed(p, key="report_hash"):
        return documents[p]

    if tamper:
        with pytest.raises(ValueError):
            require_context_baselines(cfg, jobs, sealed, digest, pins)
    else:
        assert require_context_baselines(cfg, jobs, sealed, digest, pins) == complete
        assert len(pins) == 3
