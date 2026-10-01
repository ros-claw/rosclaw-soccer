"""Check serial and parallel reconstruction produced the same learned policy.

The two manifests legitimately bind different entry-point hashes. They must
still contain exactly the same measured arrays, records and numeric policy.
This comparison is not another generation, physical rollout, or promotion.
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.output_memory_step_motor import validate_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def equal_policy_except_bank_binding(a: dict[str, Any], b: dict[str, Any]) -> bool:
    left, right = copy.deepcopy(a), copy.deepcopy(b)
    for value in (left, right):
        value.pop("model_hash")
        value["learning_receipt"].pop("physical_batch_hash")
    return left == right


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("serial-root", "parallel-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    models, manifests, banks = [], [], []
    for root in (args.serial_root, args.parallel_root):
        model = json.loads((root / "model.json").read_text())
        validate_model(model)
        manifest = _sealed(root / "rollout_manifest.json")
        path = root / "rollouts.npz"
        if (
            manifest["schema"] != "soccer.rsi.output_memory_on_policy_bank.v1"
            or manifest["partition"] != "TRAIN_CONSUMED"
            or manifest["parent_model_hash"] != model["learning_receipt"]["learner_parent_hash"]
            or manifest["report_hash"] != model["learning_receipt"]["physical_batch_hash"]
            or manifest["data_hash"] != hash_bytes(path.read_bytes())
            or any(
                obj.get(k) is not False
                for obj in (model, manifest, model["learning_receipt"])
                for k in ("promotion_authorized", "hardware_authorized")
            )
        ):
            raise ValueError("sealed current-parent simulation learning banks required")
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {k: loaded[k] for k in loaded.files}
        models.append(model)
        manifests.append(manifest)
        banks.append(arrays)
    a, b = manifests
    if any(
        a[k] != b[k]
        for k in (
            "source_summary_hash",
            "parent_model_hash",
            "records",
            "physical_rollout_count",
            "frame_sample_count",
            "independent_contexts",
        )
    ):
        raise ValueError("parallel learning changed the physical bank identity")
    left, right = banks
    if left.keys() != right.keys() or any(
        left[k].dtype != right[k].dtype or not np.array_equal(left[k], right[k]) for k in left
    ):
        raise ValueError("parallel learning changed actual training arrays")
    if not equal_policy_except_bank_binding(*models):
        raise ValueError("same data/optimizer did not produce the identical numeric policy")
    report = dict(
        schema="soccer.rsi.parallel_learning_equivalence.v1",
        parent_model_hash=a["parent_model_hash"],
        source_summary_hash=a["source_summary_hash"],
        model_hashes=[m["model_hash"] for m in models],
        manifest_hashes=[m["report_hash"] for m in manifests],
        physical_rollout_count=a["physical_rollout_count"],
        frame_sample_count=a["frame_sample_count"],
        arrays_bitwise_equal=True,
        policies_equal_except_manifest_binding=True,
        data_hashes=[m["data_hash"] for m in manifests],
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        physical_executions_added=0,
        distinct_learning_generations_added=0,
        qualification="LEARNING_EQUIVALENCE_ONLY_NOT_PHYSICAL_GAIN",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
