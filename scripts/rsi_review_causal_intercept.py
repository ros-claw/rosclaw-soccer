"""Read back all four actual SIM intercepts, including fresh neural inference.

Does not advance physics and cannot certify independent physical execution.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean

ROOT = Path("/code/rosclaw/soccer_native_evidence")
CASES = {
    "yminus-vminus": "4ec6d8bafbc95869b2942d9c66ca458888d8080d44ef712ad770539a2c994314",
    "yminus-vplus": "82788a5612ab6d0fe9f00460989471b1f4354a3945298533e23cb4ea60480b39",
    "yplus-vminus": "247dd4a3e080824e8b9e45036c4739a379983cc7c181dcb1ff9aa79802f08dd8",
    "yplus-vplus": "bb5d27c2cd8c4fbacdc72e4ccccfa5918b73a9de8996b97c419b960107626483",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    if args.output.exists():
        raise ValueError("unused output required")
    reference = ROOT / "rsi-cftrack-predicted-ball-h300-20261027/complete.json"
    if (
        digest(reference)
        != "sha256:75612fbae5c1e4df0741c16b4a5000473aac829f919adbb5367027dce75cb151"
    ):
        raise ValueError("exact source pins required")
    pins = json.loads(reference.read_text())["input_file_hashes"]
    paths = [Path(p) for p in pins if p.endswith("/policy/omnicontact/model/kick_50k.onnx")]
    if len(paths) != 1 or paths[0].is_symlink() or digest(paths[0]) != pins[str(paths[0])]:
        raise ValueError("exact pinned inference model required")
    session = ort.InferenceSession(str(paths[0]), providers=["CPUExecutionProvider"])
    scene = ROOT / "rsi-paid-torch-native-full16-20261021/cached-standard/declared-scene.xml"
    model = mujoco.MjModel.from_xml_string(shared_native_scene(scene.read_text(), prefixes=("a/",)))
    if (
        compiled_model_hash(model)
        != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("exact compiled simulation model required")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    args.output.mkdir()
    write_once(
        args.output / "declaration.json",
        dict(
            cases=CASES,
            source_head=args.expected_head,
            neural_calls_max=972,
            independent_native_steps=0,
            model_hash=digest(paths[0]),
            physical_force_is_contact_authority=True,
            hardware_authorized=False,
        ),
    )
    records = {}
    for case, h in CASES.items():
        root = ROOT / f"rsi-causal-intercept-{case}-20261027"
        report_path = root / "complete.json"
        if report_path.is_symlink() or digest(report_path) != "sha256:" + h:
            raise ValueError("exact original actual report required")
        report = json.loads(report_path.read_text())
        path = root / "actual.npz"
        if (
            report["status"] != "CAUSAL_INTERCEPT_ACTUAL_TRAINING_NOT_PROMOTED"
            or report["training_case"] != case
            or report["failure"] is not None
            or path.is_symlink()
            or digest(path) != report["actual_artifact_hash"]
        ):
            raise ValueError("complete exact original actual trace required")
        with np.load(path, allow_pickle=False) as f:
            a = {k: f[k].copy() for k in f.files}
        if (
            a["qpos"].shape != (3000, model.nq)
            or a["ball_forces"].shape != (3000, 1, 3)
            or a["neural_obs"].shape != (243, 1, 1244)
            or a["neural_actions"].shape != (243, 1, 29)
        ):
            raise ValueError("complete exact arrays required")
        for i in range(243):
            predicted = session.run(None, {"obs": a["neural_obs"][i]})[0]
            if not np.array_equal(predicted, a["neural_actions"][i]):
                raise ValueError(f"independent inference mismatch {case}/{i}")
        q = a["qpos"].reshape(300, 10, model.nq)
        force = a["ball_forces"].reshape(300, 10, 1, 3).max(axis=1)
        force6 = np.zeros((300, 1, 6), np.float64)
        force6[:, :, :3] = force
        outcome = label_first_touch(
            dict(
                force_n=force6,
                pelvis_z_per_substep_m=q[:, :, lo.rq + 2][:, None, :],
                ball_position_after_step_m=q[:, -1, lo.bq : lo.bq + 3][:, None, :],
            )
        ).to_dict()
        if outcome != report["outcome"]:
            raise ValueError("force-derived label mismatch")
        records[case] = dict(
            source_hash="sha256:" + h,
            neural_readback_exact=True,
            neural_calls=243,
            outcome=outcome,
            parent_outcome=report["parent_outcome"],
            first_contact_reference_phase=report["first_contact_reference_phase"],
        )
        write_once(args.output / f"{case}.json", records[case])
    write_once(
        args.output / "complete.json",
        dict(
            status="CAUSAL_INTERCEPT_NEURAL_FORCE_READBACK_NOT_EXECUTION",
            records=records,
            neural_calls=972,
            independent_native_steps=0,
            phase_copied_from_source_not_independently_verified=True,
            task_promoted=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
