"""Actual shared-world zero-identity pilot; no learning or action playback.

Six predeclared consumed TRAIN sentinels, two new executions each. The original
factual worlds are authenticated comparisons only, never simulation inputs.
An optional second process uses the independent manual native clock.
"""

import argparse
import json
from pathlib import Path

INDICES = (0, 6, 24, 32, 60, 64)
SOURCE = "RECEIVING_INITIALIZATION_NATIVE_IDENTITY_PENDING_OWN"
OWN = "RECEIVING_INITIALIZATION_NATIVE_IDENTITY_ALL_OWN_MATCH_NOT_TRAINED"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in (
        "experience-root",
        "own-root",
        "training-root",
        "factual-source-root",
        "factual-own-root",
        "initialization-root",
        "output",
    ):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "own-hash",
        "training-hash",
        "factual-own-hash",
        "initialization-hash",
        "expected-head",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--audit-source-root", type=Path)
    p.add_argument("--producer-head")
    a = p.parse_args()
    if (a.audit_source_root is None) != (a.producer_head is None):
        p.error("exact independent producer root/head pair required")
    a.state_id = 26
    import numpy as np
    import torch

    from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
    from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor
    from rosclaw_soccer.rsi.continuing_refinement_contract import KIND, select_parallel_experience
    from rosclaw_soccer.rsi.receiving_initialization_protocol import admit_motor
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import file_hash
    from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_learned_refinement_common import admit, open_models
    from scripts.rsi_native_context_recovery_common import check_pins
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused once-only native qualification output required")
    admitted = admit(a)
    pins = admitted[1]

    def merge(report):
        for key, value in report["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting authenticated ancestry")
            pins[key] = value

    def read(root, expected=None):
        path = root / "complete.json"
        if root.is_symlink() or path.is_symlink() or (root / "failed.json").exists():
            raise ValueError("closed ordinary successful evidence required")
        digest = file_hash(path)
        if expected is not None and digest != expected:
            raise ValueError("exact predeclared completed receipt required")
        result = json.loads(path.read_text())
        merge(result)
        pins[str(path)] = digest
        return result

    factual = read(a.factual_source_root)
    factual_own = read(a.factual_own_root, a.factual_own_hash)
    select_parallel_experience(
        factual,
        factual_own,
        source_hash=pins[str(a.factual_source_root / "complete.json")],
        training_hash=a.training_hash,
        state=26,
        full_recovery=True,
    )
    trained = json.loads((a.training_root / "complete.json").read_text())
    if (
        trained["checkpoint_kind"] != KIND
        or trained["config"]["generation_index"] != 2
        or trained["cumulative_actor_steps"] != 12
        or trained["cumulative_critic_steps"] != 12
    ):
        raise ValueError("fixed actual A generation2 theta12 required")
    init = read(a.initialization_root, a.initialization_hash)
    if (
        init["status"] != "WHOLE_RECEIVING_MLP_RECORDED_INITIALIZATION_QUALIFIED_NOT_TRAINED"
        or init["optimizer_updates"] != 0
        or init["new_physics_steps"] != 0
        or init["actual_recorded_observations"] != 37344
    ):
        raise ValueError("exact closed untrained initialization qualification required")
    cp = a.initialization_root / "untrained-initialization.pt"
    if cp.is_symlink() or file_hash(cp) != init["checkpoint_hash"]:
        raise ValueError("exact safely loaded untrained checkpoint required")
    pins[str(cp)] = init["checkpoint_hash"]
    facts = [factual["records"][i] for i in INDICES]

    def raw_pins(root, records):
        for row in records:
            for art in row["artifacts"].values():
                path = root / art["path"]
                if (
                    path.is_symlink()
                    or path.resolve().parent != (root / f"trial{row['index']}").resolve()
                ):
                    raise ValueError("confined immutable raw primitives required")
                if file_hash(path) != art["hash"]:
                    raise ValueError("actual raw comparison primitives changed")
                pins[str(path)] = art["hash"]

    raw_pins(a.factual_source_root, facts)
    source = None
    if a.audit_source_root is not None:
        source = read(a.audit_source_root)
        if (
            source["status"] != SOURCE
            or source["source_head"] != a.producer_head
            or source["initialization_hash"] != a.initialization_hash
            or source["training_hash"] != a.training_hash
            or source["source_case_indices"] != list(INDICES)
            or len(source["records"]) != 12
        ):
            raise ValueError("exact same receiving initialization producer required")
        raw_pins(a.audit_source_root, source["records"])
    source_pins(pins)
    check_pins(pins)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    setup, _, actor, courses, preflight = open_models(a, admitted)
    if any(row["course"] not in courses for row in facts):
        raise ValueError("all three declared courses must pass preflight before execution")
    params = VelocityPolicyParameters.load(
        setup.velocity_config, expected_hash=pins[str(setup.velocity_config)]
    )
    prior = VelocityPriorActor.from_onnx(
        setup.velocity_path, expected_hash=pins[str(setup.velocity_path)], reference_features=14
    )
    motor = AnchoredReceivingMotor(prior, torch.tensor(params.action_scale, dtype=torch.float32))
    state = torch.load(cp, map_location="cpu", weights_only=True)
    if (
        set(state)
        != {
            "schema",
            "actor",
            "velocity_model_hash",
            "velocity_config_hash",
            "context_features",
            "optimizer_updates",
            "trained",
        }
        or state["schema"] != "soccer.receiver_whole_mlp_initialization.v1"
        or state["velocity_model_hash"] != pins[str(setup.velocity_path)]
        or state["velocity_config_hash"] != pins[str(setup.velocity_config)]
        or state["context_features"] != 14
        or state["optimizer_updates"] != 0
        or state["trained"] is not False
    ):
        raise ValueError("exact untrained body codec checkpoint required")
    expected = motor.state_dict()
    if expected.keys() != state["actor"].keys() or any(
        not torch.equal(value, state["actor"][key]) for key, value in expected.items()
    ):
        raise ValueError("whole initialization must match original pinned ONNX and fixed cues")
    motor.load_state_dict(state["actor"], strict=True)
    motor.eval().requires_grad_(False)
    admit_motor(motor)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            source_case_indices=list(INDICES),
            repeats=2,
            input_file_hashes=pins,
            initialization_hash=a.initialization_hash,
            maximum_native_steps=48000,
            maximum_neural_calls=19200,
            preflight=preflight,
            optimizer_updates=0,
            hardware_authorized=False,
        ),
    )
    counts, records = dict(native_steps=0, actor_calls=0), []
    try:
        previous = None
        for index in range(12):
            fact = facts[index // 2]
            common = dict(
                output=a.output,
                index=index,
                counts=counts,
                emit_course_rows=False,
                sampling_seed=fact["sampling_seed"],
                sampling_profile="contact-v1",
                sampling_include_recovery=True,
                receiver_motor=motor,
            )
            if source is None:
                from scripts.rsi_pass_transfer_trial import collect_trial

                row = collect_trial(setup, actor, fact["course"], **common)
                # Compare ONLY AFTER the new body's own NN actions were executed.
                # Never send these stored actions, states or hidden histories to physics.
                for key in ("initial", "physical", "A", "B", "sampling"):
                    with (
                        np.load(
                            a.output / row["artifacts"][key]["path"], allow_pickle=False
                        ) as new,
                        np.load(
                            a.factual_source_root / fact["artifacts"][key]["path"],
                            allow_pickle=False,
                        ) as old,
                    ):
                        if not set(old.files) <= set(new.files) or any(
                            not np.array_equal(new[k], old[k]) for k in old.files
                        ):
                            raise ValueError(f"actual zero identity failed: trial{index}/{key}")
                if row["outcome"] != fact["outcome"] or row["A_recovery"] != fact["A_recovery"]:
                    raise ValueError("physical task/recovery identity changed")
            else:
                from scripts.rsi_pass_transfer_replay_trial import replay_trial

                claim = {
                    k: v for k, v in source["records"][index].items() if k != "source_case_index"
                }
                row, previous = replay_trial(
                    setup,
                    actor,
                    fact["course"],
                    claim,
                    raw_root=a.audit_source_root,
                    prior_initial=previous,
                    **common,
                )
            row["source_case_index"] = INDICES[index // 2]
            records.append(row)
            write_once(a.output / f"record{index}.json", row)
        if counts != dict(
            native_steps=sum(r["native_steps"] for r in records),
            actor_calls=sum(r["actor_calls"] for r in records),
        ):
            raise ValueError("all charged physical and 4-network controls required")
        if source is not None and (
            records != [{k: v for k, v in r.items() if k != "artifacts"} for r in source["records"]]
            or counts != source["counts"]
        ):
            raise ValueError("all independent own network/physical results must match")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status=SOURCE if source is None else OWN,
                source_head=a.expected_head,
                source_case_indices=list(INDICES),
                records=records,
                counts=counts,
                input_file_hashes=pins,
                initialization_hash=a.initialization_hash,
                training_hash=a.training_hash,
                optimizer_updates=0,
                source_hash=None
                if source is None
                else file_hash(a.audit_source_root / "complete.json"),
                independent_whole_world_replay=source is not None,
                all_actual_native_zero_identity=True,
                neural_learning_proven=False,
                Fresh_verified=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                counts=counts,
                completed_records=len(records),
                incomplete_world_native_steps_unknown=True,
                reserved_native_step_ceiling=48000,
                optimizer_updates=0,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
