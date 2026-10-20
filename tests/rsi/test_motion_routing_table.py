from pathlib import Path

import numpy as np

from scripts.rsi_motion_portfolio_common import training_table


def test_routing_table_never_reads_validation_labels_and_prefers_old_success(tmp_path):
    success_sets = [{0, 10}, {2, 10}, {12, 19}, {15}, {4}]
    cases = [{} for _ in range(5)]
    bd = {"rows": [{"split": "train" if i < 32 else "validation"} for i in range(40)]}
    for i in range(40):
        if i < 32:
            features = np.zeros((300, 108), dtype=np.float32)
            features[:, 1], features[:, 2] = 0.75, 1.0
            features[20, 70] = i / 100
            path = tmp_path / f"case{i}.npz"
            np.savez_compressed(path, student_features=features)
        else:
            path = Path("must-not-be-read-validation")
        for j, successful in enumerate(success_sets):
            row = {"score": {"qualified": i in successful}} if i < 32 else {}
            cases[j][i] = (row, path)
    x, success, targets, indices = training_table(bd, cases)
    assert x.shape == (32, 8) and success.shape == (32, 5) and indices == list(range(32))
    assert targets[0] == targets[10] == 0
    assert (targets[2], targets[12], targets[19], targets[15], targets[4]) == (1, 2, 2, 3, 4)
    assert success.any(1).sum() == 7
