import pytest

from scripts.rsi_mjlab_stage1_four_gpu import validate_gpu_mapping, validate_ranks


def test_ranks_require_four_unique_ordered_local_devices():
    env = dict(
        WORLD_SIZE="4",
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        RANK="2",
        LOCAL_RANK="2",
    )
    assert validate_ranks(env) == (2, 2)
    for changes in (
        {"WORLD_SIZE": "1"},
        {"LOCAL_RANK": "0"},
        {"RANK": "4"},
        {"CUDA_VISIBLE_DEVICES": "0,1"},
        {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"},
    ):
        with pytest.raises(ValueError):
            validate_ranks(env | changes)


def test_cuda_mapping_rejects_duplicate_or_reordered_devices():
    gpu_map = {i: f"GPU-{i}" for i in range(4)}
    validate_gpu_mapping(gpu_map, [gpu_map[i] for i in range(4)])
    with pytest.raises(ValueError):
        validate_gpu_mapping(gpu_map, [gpu_map[i] for i in (1, 0, 2, 3)])
    with pytest.raises(ValueError):
        validate_gpu_mapping({0: "a", 1: "a", 2: "c", 3: "d"}, ["a", "a", "c", "d"])
