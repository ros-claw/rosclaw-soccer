"""Diagnose rejected actor on identical old measured states; never step physics."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "inputs",
        "core-source",
        "output",
        "exam",
        "review",
        "anchor-exam",
        "anchor-audit",
        "policy",
        "anchor",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "expected-head",
        "expected-inputs-hash",
        "expected-review-hash",
        "expected-anchor-audit-hash",
        "expected-policy-hash",
        "expected-anchor-hash",
        "review-unit",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_motion_portfolio_common import admit
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    spec, bd, pins, _ = admit(a)
    wait_for_terminal_collection(a.review_unit, wait=True)
    for path, h in (
        (a.review / "complete.json", a.expected_review_hash),
        (a.anchor_audit / "complete.json", a.expected_anchor_audit_hash),
        (a.policy, a.expected_policy_hash),
        (a.anchor, a.expected_anchor_hash),
    ):
        if digest(path) != h:
            raise ValueError("exact authenticated actor/evidence identity required")
        pins[str(path)] = h
    review = json.loads((a.review / "complete.json").read_text())
    audit = json.loads((a.anchor_audit / "complete.json").read_text())
    if (
        review["status"] != "GROUPED_ACTOR_ALL40_PRIMITIVES_AUTHENTICATED"
        or review["candidate_hash"] != a.expected_policy_hash
        or review["verdict"]["necessary_research_growth"] is not False
        or audit["raw_outcomes_recomputed"] is not True
        or audit["adam_state_and_rng_tensor_exact"] is not True
    ):
        raise ValueError("diagnosis requires actual rejected all40 and audited anchor")
    for evidence in (audit, review):
        for k, h in evidence["input_file_hashes"].items():
            if k in pins and pins[k] != h:
                raise ValueError("conflicting immutable evidence")
            pins[k] = h
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            contexts=40,
            same_old_observations=True,
            no_parameter_updates=True,
            new_native_steps=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    try:
        import mujoco

        from scripts.rsi_audit_online_kinematic_motion import read_phase

        world = mujoco.MjModel.from_binary_path(spec["world"])
        source = "37174508238b36e08983f0002f2ef61c79b82b89"
        old = read_phase(
            SimpleNamespace(seed=29, expected_pipeline_source=source),
            bd,
            pins,
            world,
            a.anchor_exam,
            "evaluate",
            0,
            a.expected_anchor_hash,
        )
        new = read_phase(
            SimpleNamespace(seed=17, expected_pipeline_source=source),
            bd,
            pins,
            world,
            a.exam,
            "evaluate",
            0,
            a.expected_policy_hash,
        )
        result = diagnose(a, spec, old, new)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="REJECTED_GROUPED_ACTOR_SAME_STATE_DRIFT_DIAGNOSED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                rows=result,
                new_native_steps=0,
                optimizer_steps=0,
                no_causal_improvement_inferred=True,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(error=repr(error), automatic_retry_authorized=False, new_native_steps=0),
        )
        raise


def diagnose(a, spec, old, new):
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.quantized_action_drift import first_physical_divergence, token_drift
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_online_kinematic_motion import load_model

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    anchor = load_model(spec, a.anchor).eval().requires_grad_(False)
    candidate = load_model(spec, a.policy).eval().requires_grad_(False)
    before, after = anchor.state_dict(), candidate.state_dict()
    for k in before:
        if not (k.startswith("adapter.") or k == "log_std") and not torch.equal(
            before[k], after[k]
        ):
            raise ValueError("diagnostic cannot conflate parent/critic/normalizer changes")
    old_rows = {
        r["course_index"]: (root, r) for root, phase in old for r in phase["result"]["results"]
    }
    new_rows = {
        r["course_index"]: (root, r) for root, phase in new for r in phase["result"]["results"]
    }
    result = []
    for i in range(40):
        old_root, old_row = old_rows[i]
        new_root, new_row = new_rows[i]
        with np.load(old_root / f"course{i:02d}" / "native.npz", allow_pickle=False) as z:
            x = torch.from_numpy(z["student_features"].copy())[None]
            actual_old_token = z["student_motion_token"].copy()
            actual_old_q = z["target_body_canonical_qpos_after_substep"].copy()
        with np.load(new_root / f"course{i:02d}" / "native.npz", allow_pickle=False) as z:
            actual_new_q = z["target_body_canonical_qpos_after_substep"].copy()
        with np.load(old_root / f"course{i:02d}" / "sampling.npz", allow_pickle=False) as z:
            recorded_mean = z["mean"].copy()
        with torch.no_grad():
            geometry = anchor.geometry(x)
            mu, std, _, _ = anchor.parameters_for_state(x, geometry=geometry)
            next_mu, next_std, _, _ = candidate.parameters_for_state(x, geometry=geometry)
            torch.testing.assert_close(mu[0], torch.from_numpy(recorded_mean), atol=1e-6, rtol=1e-5)

            def quantize(t):
                return (
                    torch.round(15.515501022338867 * torch.tanh(t + 0.03223699703812599) - 0.5) / 16
                )[0].numpy()

            old_token, next_token = quantize(mu), quantize(next_mu)
            kl = (
                torch.log(next_std / std)
                + (std.square() + (mu - next_mu).square()) / (2 * next_std.square())
                - 0.5
            ).sum(-1)[0, 20:]
        row = dict(
            course_index=i,
            split=old_row["split"],
            anchor_qualified=old_row["score"]["qualified"],
            candidate_qualified=new_row["score"]["qualified"],
            same_state_token_drift=token_drift(old_token, next_token),
            anchor_batch_vs_native_token_drift=token_drift(actual_old_token, old_token),
            mean_gaussian_kl_on_old_deterministic_states=float(kl.mean()),
            max_gaussian_kl_on_old_deterministic_states=float(kl.max()),
            maximum_mean_delta=float((mu - next_mu).abs().max()),
            first_actual_native_substep_qpos_divergence=first_physical_divergence(
                actual_old_q, actual_new_q
            ),
        )
        write_once(a.output / f"course{i:02d}.json", row)
        result.append(row)
        print("SAME_STATE_ACTION_DRIFT", row, flush=True)
    return result


if __name__ == "__main__":
    main()
