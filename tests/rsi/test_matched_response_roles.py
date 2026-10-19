import pytest

from rosclaw_soccer.rsi.matched_response_roles import admit_role


def test_retention_overrides_shared_training_ancestry():
    row = dict(family="old252", context_index=12, source_cluster=0, split="train")
    admitted = admit_role(row, "CONSUMED_ORIGINAL_RETENTION")
    assert admitted["split"] == "retention"
    assert admitted["gradient_eligible"] is False
    assert admitted["legacy_split_NOT_authoritative"] == "train"
    assert row["split"] == "train"


@pytest.mark.parametrize("cluster,split", [(0, "train"), (1, "validation"), (2, "test")])
def test_local_ancestry_group(cluster, split):
    row = dict(family="old252", context_index=cluster * 4, source_cluster=cluster)
    result = admit_role(row, "CONSUMED_LOCAL_DEVELOPMENT")
    assert result["split"] == split
    assert result["gradient_eligible"] == (split == "train")


@pytest.mark.parametrize("cluster,split", [(0, "train"), (6, "validation"), (8, "test")])
def test_public_group(cluster, split):
    row = dict(family="public40", context_index=3, source_cluster=cluster)
    assert admit_role(row, None)["split"] == split


@pytest.mark.parametrize("partition", [None, "CONSUMED_LOCAL_DEVELOPMENT", "train"])
def test_original_retention_cannot_be_relabelled(partition):
    row = dict(family="old252", context_index=12, source_cluster=0)
    with pytest.raises(ValueError):
        admit_role(row, partition)


def test_unknown_family_or_boolean_cluster_rejected():
    with pytest.raises(ValueError):
        admit_role(dict(family="other", context_index=0, source_cluster=0), None)
    with pytest.raises(ValueError):
        admit_role(dict(family="public40", context_index=0, source_cluster=True), None)
