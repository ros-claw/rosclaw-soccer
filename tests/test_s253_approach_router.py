import json

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.approach_router import G1ApproachRouter
from rosclaw_soccer.sim.contracts import hash_bytes

BODY = "sha256:" + "a" * 64
LIBRARY = "sha256:" + "b" * 64


def artifact(tmp_path):
    shapes = {
        "0.weight": (64, 4),
        "0.bias": (64,),
        "2.weight": (64, 64),
        "2.bias": (64,),
        "4.weight": (3, 64),
        "4.bias": (3,),
    }
    arrays = {k: np.zeros(v, dtype=np.float32) for k, v in shapes.items()}
    arrays["4.bias"][:] = [-1, 2, 0]
    weights = tmp_path / "router.npz"
    np.savez_compressed(weights, **arrays)
    metadata = dict(
        schema="rosclaw_soccer.g1_approach_router.v1",
        activation_ceiling="SIM_ONLY",
        body_actor_hash=BODY,
        reference_library_hash=LIBRARY,
        reference_indices=[14, 40, 65],
        feature_scales=[0.08, 0.08, 0.4, 0.32],
        feature_lower=[-0.08, -0.08, -0.05, -0.32],
        feature_upper=[0.08, 0.08, 0.45, 0.08],
        weights_hash=hash_bytes(weights.read_bytes()),
    )
    path = tmp_path / "router.json"
    path.write_text(json.dumps(metadata))
    return path, metadata


def load(path, **kwargs):
    return G1ApproachRouter(
        path,
        expected_body_actor_hash=kwargs.get("body", BODY),
        expected_reference_library_hash=LIBRARY,
    )


def test_content_bound_data_only_proposal(tmp_path):
    path, _ = artifact(tmp_path)
    router = load(path)
    result = router.propose(ball_offset_xy_m=(0.0, 0.0), ball_velocity_xy_mps=(0.2, -0.1))
    assert result.reference_index == 40
    assert result.predicted_utility == 2.0  # utility, not probability
    assert result.activation_ceiling == "SIM_ONLY"
    with pytest.raises(ValueError):
        router._parameters["4.bias"][0] = 100.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 0.081])
def test_bad_or_out_of_domain_observation_is_not_clipped_into_qualification(tmp_path, value):
    path, _ = artifact(tmp_path)
    with pytest.raises(ValueError):
        load(path).propose(ball_offset_xy_m=(value, 0.0), ball_velocity_xy_mps=(0.0, 0.0))


def test_body_binding_and_numeric_weights_tamper_fail(tmp_path):
    path, _ = artifact(tmp_path)
    with pytest.raises(ValueError, match="body/reference"):
        load(path, body="sha256:" + "c" * 64)
    path.with_suffix(".npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        load(path)


def test_small_archive_cannot_request_huge_array_allocation(tmp_path):
    import io
    import zipfile

    path, metadata = artifact(tmp_path)
    weights = path.with_suffix(".npz")
    with zipfile.ZipFile(weights) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, dict(descr="<f4", fortran_order=False, shape=(10**12, 4))
    )
    entries["0.weight.npy"] = header.getvalue()
    with zipfile.ZipFile(weights, "w") as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    metadata["weights_hash"] = hash_bytes(weights.read_bytes())
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="bounded shape"):
        load(path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("activation_ceiling", "REAL"),
        ("reference_indices", [1, 1]),
        ("reference_indices", [False, 1]),
        ("feature_scales", [0, 1, 1, 1]),
    ],
)
def test_manifest_cannot_expand_boundary_or_alias_references(tmp_path, key, value):
    path, metadata = artifact(tmp_path)
    metadata[key] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load(path)
