"""Finite joint precision/speed teacher collection OR whole private native audit.

Serial CPU MuJoCo research only. Reuse the qualified baseline, execute38 new
worlds per pilot, retain every ordinary failure, no optimizer or Fresh access.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("expert-own-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("expert-own-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--state-id", type=int, choices=(25, 26), required=True)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--source-root", type=Path)
    p.add_argument("--source-hash")
    p.add_argument("--producer-head")
    a = p.parse_args()
    if a.producer_head is not None and a.source_root is None:
        p.error("producer head requires an audit source")
    if (a.source_root is None) != (a.source_hash is None) or (
        a.source_root is not None and a.preflight_only
    ):
        p.error("audit requires both source root/hash, not preflight-only")
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_joint_delivery_pilot_common import admit, local_preflight
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused once-only pilot/audit output required")
    from rosclaw_soccer.rsi.joint_delivery_teacher import pilot_config, verify_pilot_search

    cfg = pilot_config(a.state_id)
    pins, scope, bank, pf, inherited_root, baseline = admit(
        a.expert_own_root, a.expert_own_hash, cfg
    )
    source = None
    if a.source_root is not None:
        sp, dp = a.source_root / "complete.json", a.source_root / "declaration.json"
        if (
            a.source_root.is_symlink()
            or any(p.is_symlink() for p in (sp, dp))
            or (a.source_root / "failed.json").exists()
            or digest(sp) != a.source_hash
        ):
            raise ValueError("exact complete nonfailed pilot source required for audit")
        source = json.loads(sp.read_text())
        source_declaration = json.loads(dp.read_text())
        if (
            source["status"] != "JOINT_DELIVERY_TEACHER38_COLLECTED_PENDING_ALL_OWN"
            or source["config"] != cfg
            or source["expert_own_hash"] != a.expert_own_hash
            or source["source_head"] != (a.producer_head or a.expected_head)
            or source["declaration_hash"] != digest(dp)
            or len(source["records"]) != 38
            or source["optimizer_updates"] != 0
            or source["input_file_hashes"] != source_declaration["input_file_hashes"]
            or source_declaration["config"] != cfg
            or source_declaration["expert_own_hash"] != a.expert_own_hash
            or source_declaration["preflight_only"] is not False
            or any(
                source[k] is not False
                for k in (
                    "hardware_authorized",
                    "promotion_authorized",
                    "goal_complete",
                    "Fresh_verified",
                    "team_game_verified",
                )
            )
        ):
            raise ValueError("same exact full predeclared38-world pilot required")
        for key, value in source["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting source/own ancestry")
            pins[key] = value
        pins.update({str(sp): a.source_hash, str(dp): digest(dp)})
        for row in source["records"]:
            for art in row["artifacts"].values():
                path = a.source_root / art["path"]
                if (
                    path.is_symlink()
                    or path.resolve().parent != (a.source_root / f"trial{row['index']}").resolve()
                    or digest(path) != art["hash"]
                ):
                    raise ValueError("all confined actual source primitives required")
                pins[str(path)] = art["hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    declared = dict(
        source_head=a.expected_head,
        input_file_hashes=pins,
        config=cfg,
        expert_own_hash=a.expert_own_hash,
        expert_own_root=str(a.expert_own_root),
        producer_hash=a.source_hash,
        preflight_only=a.preflight_only,
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
    )
    write_once(a.output / "declaration.json", declared)
    counts = dict(native_steps=0, actor_calls=0)
    records = []
    try:
        import numpy as np

        from scripts.rsi_self_teacher_search_common import compare_common, open_setup

        setup, actor = open_setup(scope, pf, pins, bank, inherited_root / "scene.xml")
        variants, preflight = local_preflight(setup, baseline, cfg, inherited_root)
        write_once(a.output / "preflight.json", preflight)
        if a.preflight_only:
            status = "JOINT_DELIVERY_TEACHER_LOCAL_AND_ORIGINAL40_PREFLIGHT_NOT_PHYSICS"
            payload = dict(physical_steps_executed=0, optimizer_updates=0)
        elif source is not None:
            from scripts.rsi_pass_transfer_replay_trial import replay_trial

            if preflight != source["preflight"]:
                raise ValueError("all independently regenerated local courses differ")
            verify_pilot_search(source, baseline, variants)
            previous = None
            for index, claim in enumerate(source["records"]):
                metadata = {k: claim[k] for k in ("tag", "prefix_identity_fields")}
                stripped = {k: v for k, v in claim.items() if k not in metadata}
                actual, previous = replay_trial(
                    setup,
                    actor,
                    claim["course"],
                    stripped,
                    raw_root=a.source_root,
                    output=a.output,
                    index=index,
                    counts=counts,
                    prior_initial=previous,
                    emit_course_rows=False,
                )
                prefix = (
                    compare_common(a.source_root, claim, inherited_root, baseline, prefix=80)
                    if index < 34
                    else None
                )
                if prefix != metadata["prefix_identity_fields"]:
                    raise ValueError("independently checked baseline prefix relation differs")
                records.append(actual | dict(tag=metadata["tag"], prefix_identity_fields=prefix))
            if counts != source["counts"]:
                raise ValueError("all actual own clock/inference counters must match")
            verify_pilot_search(source | dict(records=records), baseline, variants)
            status = "JOINT_DELIVERY_TEACHER38_ALL_WHOLE_OWN_NATIVE_MATCH"
            payload = dict(
                records=records,
                counts=counts,
                source_hash=a.source_hash,
                selected=source["selected"],
                optimizer_updates=0,
                independent_whole_world_replay=True,
            )
        else:
            from rosclaw_soccer.rsi.joint_delivery_teacher import joint_delivery_rank
            from rosclaw_soccer.rsi.stroke_search import sample_generation, update_elites
            from scripts.rsi_pass_transfer_trial import collect_trial

            def run(tag, course, latent):
                if len(records) >= cfg["maximum_worlds"]:
                    raise ValueError("declared finite38-world pilot exhausted")
                c = course | dict(assisted=True, latent=latent.tolist())
                row = collect_trial(
                    setup,
                    actor,
                    c,
                    output=a.output,
                    index=len(records),
                    counts=counts,
                    emit_course_rows=False,
                )
                records.append(row)  # Preserve charged work before any subsequent check.
                row.update(
                    tag=tag,
                    prefix_identity_fields=compare_common(
                        a.output, row, inherited_root, baseline, prefix=80
                    )
                    if row["index"] < 34
                    else None,
                )
                write_once(a.output / f"record{row['index']}.json", row)
                return dict(tag=tag, latent=latent.copy(), row=row)

            incumbent = dict(tag="inherited", latent=np.zeros((2, 6)), row=baseline)
            mean, std, history = np.zeros((2, 6)), np.full((2, 6), 0.4), []
            rng = np.random.default_rng(cfg["rng_seed"])
            for g in range(2):
                samples, draws = sample_generation(rng, mean, std)
                values = [incumbent] + [
                    run(f"g{g}-c{k}", baseline["course"], z) for k, z in enumerate(samples)
                ]
                ranks = [joint_delivery_rank(v["row"]) for v in values]
                updated, spread, elite = update_elites([v["latent"] for v in values], ranks)
                incumbent = values[max(range(17), key=lambda k: (ranks[k], -k))]
                history.append(
                    dict(
                        generation=g,
                        mean_before=mean.tolist(),
                        std_before=std.tolist(),
                        draws=draws,
                        tags=[v["tag"] for v in values],
                        ranks=[list(r) for r in ranks],
                        elite=elite,
                        mean_after=updated.tolist(),
                        std_after=spread.tolist(),
                        selected=incumbent["tag"],
                    )
                )
                mean, std = updated, spread
            for repeat in range(2):
                run(f"selected-repeat{repeat}", baseline["course"], incumbent["latent"])
            for index, variant in enumerate(variants):
                for repeat in range(2):
                    run(f"validation{index}-repeat{repeat}", variant, incumbent["latent"])
            selected = dict(
                tag=incumbent["tag"],
                latent=incumbent["latent"].tolist(),
                rank=list(joint_delivery_rank(incumbent["row"])),
                outcome=incumbent["row"]["outcome"],
            )
            payload = dict(
                records=records,
                counts=counts,
                generations=history,
                selected=selected,
                preflight=preflight,
                optimizer_updates=0,
            )
            verify_pilot_search(payload | dict(config=cfg), baseline, variants)
            status = "JOINT_DELIVERY_TEACHER38_COLLECTED_PENDING_ALL_OWN"
        if (
            counts["native_steps"] > cfg["maximum_native_steps"]
            or counts["actor_calls"] > cfg["maximum_actor_calls"]
        ):
            raise ValueError("unchanged actual physics/inference budget exceeded")
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                **declared,
                status=status,
                declaration_hash=digest(a.output / "declaration.json"),
                **payload,
                Fresh_verified=False,
                team_game_verified=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                counts=counts,
                records=records,
                optimizer_updates=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
