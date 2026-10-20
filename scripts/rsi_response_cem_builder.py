"""Own-response CEM builder; archives assert anchors, never choose samples."""


def build_candidates(predict, rows, prefix, source, item, folder):
    import numpy as np

    from rosclaw_soccer.rsi.contact_response_cem import (
        SamplingExhausted,
        elite_update,
        response_merit,
        sample_four,
    )
    from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest

    root, complete = source
    # Historical data is used ONLY to assert same-state anchors are unchanged.
    old = complete["results"][item.variant_index]["results"][1]["candidates"]
    if len(old) != 6:
        raise ValueError("all six original anchors required")
    for law in (0, 3, 4):
        for offset in ((0.0, 0.0, 0.0), (0.0, -0.025, -0.025)):
            index = len(rows)
            future = predict(law, offset)
            record = old[index]
            if record["law"]["feedback_index"] != law or record[
                "contact_reference_offset_m"
            ] != list(offset):
                raise ValueError("historical anchor identity differs")
            p = root / f"variant{item.variant_index}" / record["artifact"]
            if digest(p) != record["artifact_hash"]:
                raise ValueError("historical anchor artifact changed")
            with np.load(p, allow_pickle=False) as a:
                if set(a.files) != set(future.arrays):
                    raise ValueError("all18 anchor arrays required")
                for k in a.files:
                    np.testing.assert_array_equal(a[k], future.arrays[k])
    prefix_hash = hash_json(
        {
            k: dict(shape=list(a.shape), dtype=a.dtype.str, bytes_hash=hash_bytes(a.tobytes()))
            for k, a in sorted(prefix.items())
        }
    )
    iterations = []
    for law_index, anchor_indices in ((0, (0, 1)), (3, (2, 3)), (4, (4, 5))):
        seed_hash = hash_json(
            dict(
                operation="response_cem_two_rounds_v1", prefix_hash=prefix_hash, law_index=law_index
            )
        )
        seed = int(seed_hash.split(":")[1][:8], 16)
        rng = np.random.default_rng(seed)
        best = max(anchor_indices, key=lambda i: (response_merit(rows[i]["score"]), -i))
        mean = np.array(rows[best]["offset_root_m"], dtype=np.float64)
        std = np.full(3, 0.015, dtype=np.float64)
        for iteration in range(2):
            try:
                samples, draws = sample_four(rng, mean, std)
            except SamplingExhausted as error:
                write_once(
                    folder / f"cem-law{law_index}-round{iteration}-sampling-failure.json",
                    dict(
                        seed=seed,
                        seed_hash=seed_hash,
                        prefix_content_hash=prefix_hash,
                        mean_m=mean.tolist(),
                        std_m=std.tolist(),
                        all_draws=error.draws,
                        native_steps_from_unexecuted_batch=0,
                        no_partial_batch_executed=True,
                        promotion_authorized=False,
                        hardware_authorized=False,
                    ),
                )
                raise
            proposal = dict(
                law_index=law_index,
                iteration=iteration,
                prefix_content_hash=prefix_hash,
                seed=seed,
                seed_hash=seed_hash,
                rng="numpy.default_rng.PCG64",
                mean_m=mean.tolist(),
                std_m=std.tolist(),
                samples=[list(x) for x in samples],
                all_draws=draws,
                retained_best_before=best,
                anchors_all18_bit_exact=True,
                merit_is_official_exam=False,
                optimizer_updates=0,
                promotion_authorized=False,
                hardware_authorized=False,
            )
            write_once(folder / f"cem-law{law_index}-round{iteration}-proposals.json", proposal)
            new_indices = []
            for sample_index, sample in enumerate(samples):
                new_indices.append(len(rows))
                predict(
                    law_index,
                    sample,
                    dict(
                        method="bounded_own_response_cem",
                        iteration=iteration,
                        sample_index=sample_index,
                        prefix_content_hash=prefix_hash,
                        seed=seed,
                    ),
                )
            pool = new_indices + [best]
            points = [tuple(rows[i]["offset_root_m"]) for i in pool]
            merits = [response_merit(rows[i]["score"]) for i in pool]
            mean, std, elite = elite_update(points, merits)
            best = max(pool, key=lambda i: (response_merit(rows[i]["score"]), -i))
            updated = dict(
                proposal,
                pool_indices=pool,
                merits=merits,
                elite_indices=[pool[i] for i in elite],
                updated_mean_m=mean.tolist(),
                updated_std_m=std.tolist(),
                retained_best_after=best,
            )
            write_once(folder / f"cem-law{law_index}-round{iteration}-complete.json", updated)
            iterations.append(updated)
    if len(rows) != 30 or len(iterations) != 6:
        raise ValueError("exact preregistered CEM budget required")
    return iterations
