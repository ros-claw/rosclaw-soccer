"""Synthetic artifacts exercise readback; these are not decoder benchmarks."""

import json

import pytest

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_read_generated_binding_benchmark import digest, readback


def write(path, value, sealed=False):
    if sealed:
        value = dict(value, report_hash=hash_json(value))
    path.write_text(json.dumps(value))
    return value


@pytest.fixture
def artifacts(tmp_path):
    original = tmp_path / "source-bytes"
    original.write_bytes(b"synthetic source")
    declaration = write(
        tmp_path / "declaration.json",
        dict(
            seeds=[30300, 30301],
            input_file_hashes={str(original): digest(original)},
            order_by_seed={
                "30300": ["external_bind", "generated_bind"],
                "30301": ["generated_bind", "external_bind"],
            },
            expected_canonical_mean_hash="mean",
            expected_numeric_prototype_hash="prototype",
        ),
        True,
    )
    write(
        tmp_path / "allocated.json",
        dict(
            original_complete_constructor_seconds=5,
            canonical_mean_hash="mean",
            numeric_prototype_hash="prototype",
        ),
    )
    rows = []
    for seed in declaration["seeds"]:
        order = declaration["order_by_seed"][str(seed)]
        times = {"external_bind": 4.0, "generated_bind": 1.0}
        for method in order:
            write(
                tmp_path / f"measured-{seed}-{method}.json",
                dict(
                    seed=seed,
                    method=method,
                    elapsed_seconds=times[method],
                    policy_hash=f"policy-{seed}",
                    whole_preview_hash=f"whole-{seed}",
                ),
            )
        rows.append(
            write(
                tmp_path / f"qualified-{seed}.json",
                dict(
                    seed=seed,
                    order=order,
                    compared_frames=300,
                    complete_documents_equal=True,
                    actions_hidden_and_sampling_transitions_bit_equal=True,
                    combined_generation_binding_seconds=times,
                    local_combined_speedup=4.0,
                    whole_preview_hash=f"whole-{seed}",
                ),
                True,
            )
        )
    write(
        tmp_path / "complete.json",
        dict(
            schema="soccer.rsi.generated_episode_binding_real_model_benchmark.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            compared_numeric_frames=600,
            original_complete_constructor_seconds=5,
            native_collector_optimization_enabled=False,
            end_to_end_speedup_claimed=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
            new_native_bodies=0,
            new_independent_physics_steps=0,
            new_optimizer_updates=0,
        ),
        True,
    )
    return tmp_path


def test_complete_readback_has_explicit_evidence_ceiling(artifacts):
    result = readback(artifacts)
    assert len(result["rows"]) == 2
    assert result["new_physical_steps"] == 0
    assert result["numeric_decoder_reexecuted_here"] is False
    assert result["native_noninterference_established"] is False


def test_changed_original_bytes_rejected(artifacts):
    (artifacts / "source-bytes").write_bytes(b"changed")
    with pytest.raises(ValueError, match="whole source"):
        readback(artifacts)


@pytest.mark.parametrize(
    "key,value",
    [
        ("elapsed_seconds", -1),
        ("whole_preview_hash", "other"),
        ("policy_hash", "other"),
        ("seed", 1),
    ],
)
def test_unsealed_measurement_must_match_sealed_pair(artifacts, key, value):
    path = artifacts / "measured-30300-generated_bind.json"
    measured = json.loads(path.read_text())
    measured[key] = value
    write(path, measured)
    with pytest.raises(ValueError):
        readback(artifacts)


def test_resealed_local_speedup_cannot_override_measured_times(artifacts):
    path = artifacts / "qualified-30300.json"
    row = json.loads(path.read_text())
    del row["report_hash"]
    row["local_combined_speedup"] = 100
    row = write(path, row, True)
    complete = json.loads((artifacts / "complete.json").read_text())
    del complete["report_hash"]
    complete["rows"][0] = row
    write(artifacts / "complete.json", complete, True)
    with pytest.raises(ValueError, match="ratio"):
        readback(artifacts)
