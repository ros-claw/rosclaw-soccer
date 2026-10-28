"""Extract native-verified receiving demonstrations; retain negative cases."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_receive_demonstration import qualify_receive_window
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import check_pins, digest
from scripts.rsi_native_fresh_common import clean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "repeat-root", "qualification-root", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    for key in ("source-hash", "declaration-hash", "qualification-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    source_path = a.source_root / "complete.json"
    repeat_path = a.repeat_root / "complete.json"
    qual_path = a.qualification_root / "complete.json"
    declaration_path = a.source_root / "declaration.json"
    repeat_declaration_path = a.repeat_root / "declaration.json"
    if (
        digest(source_path) != a.source_hash
        or digest(repeat_path) != a.source_hash
        or digest(qual_path) != a.qualification_hash
        or digest(declaration_path) != a.declaration_hash
        or digest(repeat_declaration_path) != a.declaration_hash
    ):
        raise ValueError("same independently rerun receipt and qualification required")
    source = json.loads(source_path.read_text())
    qual = json.loads(qual_path.read_text())
    declaration = json.loads(declaration_path.read_text())
    if (
        source.get("status") != "MOTIONDECODE_INCOMING_SINGLE_ACTOR_PROBE_NOT_SKILL"
        or source.get("ankle_rate_filter") is not True
        or source.get("training") is not False
        or source.get("hardware_authorized") is not False
        or source.get("promotion_authorized") is not False
        or source.get("worlds") != 32
        or source.get("receive_window") != 60
        or qual.get("status") != "NATIVE_CONTACT24_COURSE_QUALIFIED_NOT_SKILL"
        or declaration.get("fixed_speed_mps") != 1.25
        or declaration.get("qualification_hash") != a.qualification_hash
        or declaration.get("reference_hash") != source.get("reference_hash")
        or declaration.get("course_binding") != source.get("course_binding")
        or declaration.get("source_head") != source.get("source_head")
    ):
        raise ValueError("unpromoted complete incoming simulation source required")
    pins = {
        str(source_path): a.source_hash,
        str(repeat_path): a.source_hash,
        str(qual_path): a.qualification_hash,
        str(declaration_path): a.declaration_hash,
        str(repeat_declaration_path): a.declaration_hash,
        str(a.source_root / "physical.npz"): source["raw_hash"],
        str(a.repeat_root / "physical.npz"): source["raw_hash"],
        str(a.source_root / "incoming-bank.npz"): source["bank_hash"],
    }
    model_paths = [
        Path(path)
        for path in qual["input_file_hashes"]
        if path.endswith("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb")
    ]
    if len(model_paths) != 1:
        raise ValueError("one qualified native model required")
    model_path = model_paths[0]
    pins[str(model_path)] = qual["input_file_hashes"][str(model_path)]
    check_pins(pins)
    model = mujoco.MjModel.from_binary_path(str(model_path))
    layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
    with np.load(a.source_root / "physical.npz", allow_pickle=False) as z:
        raw = {k: z[k].copy() for k in z.files}
    with np.load(a.source_root / "incoming-bank.npz", allow_pickle=False) as z:
        bank = {k: z[k].copy() for k in z.files}
    if (
        raw["valid"].shape != (60, 32)
        or raw["observation_before"].shape != (60, 32, 177)
        or raw["action"].shape != (60, 32, 29)
        or not np.isfinite(raw["observation_before"]).all()
        or not np.isfinite(raw["action"]).all()
    ):
        raise ValueError("complete finite actual teacher policy history required")
    subsequent_valid = raw["valid"][1:]
    if not np.array_equal(
        raw["observation_before"][1:, :, 69:98][subsequent_valid],
        raw["action"][:-1][subsequent_valid],
    ):
        raise ValueError("teacher observations must contain previous actual action")
    other = [i for i in range(29) if i not in (4, 5, 10, 11)]
    if not np.array_equal(
        raw["action"][:, :, other][raw["valid"]],
        raw["nominal_action"][:, :, other][raw["valid"]],
    ):
        raise ValueError("teacher cannot alter declared non-ankle policy axes")
    results = []
    for world, index in enumerate(raw["bank_index"]):
        result = qualify_receive_window(
            qpos=raw["qpos"][:, world],
            qvel=raw["qvel"][:, world],
            forces=raw["forces"][:, world],
            valid=raw["valid"][:, world],
            terminated=raw["terminated"][:, world],
            ball_q=layout.bq,
            ball_v=layout.bv,
            root_q=layout.rq,
            root_v=layout.rv,
        )
        results.append(
            dict(
                world=world,
                bank_index=int(index),
                source_id=int(bank["source_id"][index]),
                **result,
            )
        )
    selected = np.array([r["qualified"] for r in results], dtype=bool)
    if not selected.any():
        raise ValueError("no strictly observed demonstration available")
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir()
    archive = a.output / "demonstrations.npz"
    np.savez_compressed(
        archive,
        observation=raw["observation_before"],
        actual_action=raw["action"],
        nominal_action=raw["nominal_action"],
        valid=raw["valid"],
        selected=selected,
        bank_index=raw["bank_index"],
    )
    write_once(
        a.output / "complete.json",
        dict(
            status="NATIVE_RECEIVE_DEMONSTRATIONS_NOT_SKILL",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            qualification_hash=a.qualification_hash,
            source_declaration_hash=a.declaration_hash,
            source_speed_mps=1.25,
            input_hashes=pins,
            dataset_hash=digest(archive),
            worlds=results,
            selected_worlds=int(selected.sum()),
            negative_worlds=int((~selected).sum()),
            same_implementation_process_repeat=True,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
