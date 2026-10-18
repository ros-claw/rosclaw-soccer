"""Read all four audited torque teacher pairs; no new simulation or training."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.pd_feedback_attribution import attribute_pd_feedback
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once


def digest(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    audit = _sealed(args.audit)
    root = Path(cfg["output_root"])
    complete = _sealed(root / "complete.json")
    if (
        args.output.exists()
        or cfg.get("use_contact_exit_torque_mpc") is not True
        or audit.get("schema") != "soccer.rsi.contact_exit_torque_mpc_terminal_readback.v1"
        or audit.get("complete_hash") != complete["report_hash"]
        or any(
            audit.get(k) is not True
            for k in (
                "whole_input_and_witness_files_verified",
                "shared_payload_logical_seals_rebuilt",
                "prior_independent_replays_authenticated",
            )
        )
        or len(complete["results"]) != 4
        or len(cfg["baseline_receipts"]) != 4
    ):
        raise ValueError("whole four-pair terminal audit and unused output required")
    import rosclaw_soccer.rsi.pd_feedback_attribution as module

    pins = {
        str(p): digest(p)
        for p in (
            args.config,
            args.audit,
            root / "complete.json",
            Path(__file__),
            Path(module.__file__),
        )
    }
    rows = []
    for item, receipt in zip(cfg["baseline_receipts"], complete["results"], strict=True):
        baseline = json.loads(Path(item["path"]).read_text())
        if (
            digest(item["path"]) != item["file_hash"]
            or baseline["receipt_hash"] != item["receipt_hash"]
            or baseline["receipt_hash"]
            != hash_json({k: v for k, v in baseline.items() if k != "receipt_hash"})
            or baseline["candidate"] != receipt["candidate"]
            or baseline["group"] != receipt["group"]
            or item["pair_id"] != receipt["pair_id"]
        ):
            raise ValueError("unchanged parent candidate and exact original pair required")
        pins[item["path"]] = item["file_hash"]
        traces, commitments = [], []
        for endpoint in (baseline, receipt):
            for name in ("trace", "commitment"):
                witness = endpoint["witnesses"][name]
                if digest(witness["path"]) != witness["sha256"]:
                    raise ValueError("actual paired source bytes changed")
                pins[witness["path"]] = witness["sha256"]
            with np.load(endpoint["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
                traces.append(dict(archive))
            commitments.append(
                json.loads(Path(endpoint["witnesses"]["commitment"]["path"]).read_text())
            )
        if (
            any(
                c["compiled_model_hash"] != cfg["mesh_task_contact_world_hash"] for c in commitments
            )
            or commitments[0]["course"] != commitments[1]["course"]
            or any(
                commitments[0]["physics"][k] != commitments[1]["physics"][k]
                for k in ("canonical_pd_kp", "canonical_pd_kd")
            )
        ):
            raise ValueError("identical paired world, course and gains required")
        measured = attribute_pd_feedback(
            *traces,
            kp=np.array(commitments[0]["physics"]["canonical_pd_kp"]),
            kd=np.array(commitments[0]["physics"]["canonical_pd_kd"]),
        )
        rows.append(dict(pair_id=item["pair_id"], attribution=measured))
    if any(digest(p) != expected for p, expected in pins.items()):
        raise ValueError("paired attribution input changed during readback")
    result = dict(
        schema="soccer.rsi.paired_pd_feedback_readback.v1",
        rows=rows,
        terminal_audit_hash=audit["report_hash"],
        input_file_hashes=pins,
        new_physical_steps=0,
        new_optimizer_updates=0,
        causal_component_intervention_performed=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print(result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
