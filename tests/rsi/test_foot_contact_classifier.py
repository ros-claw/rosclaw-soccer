import pytest

from scripts.rsi_train_foot_contact_classifier import labels_and_groups


def test_mixed_contact_is_nonfoot_and_no_contact_not_a_training_example():
    rows = [
        dict(receiver_contact=True, feature_index=0, context=2, first_contact_force_categories=[0]),
        dict(
            receiver_contact=True, feature_index=1, context=3, first_contact_force_categories=[1, 2]
        ),
        dict(receiver_contact=False, feature_index=None, context=4),
    ]
    y, groups = labels_and_groups(rows, 2)
    assert y.tolist() == [0, 1]
    assert groups.tolist() == [2, 3]
    rows[2]["feature_index"] = 2
    with pytest.raises(ValueError):
        labels_and_groups(rows, 2)
