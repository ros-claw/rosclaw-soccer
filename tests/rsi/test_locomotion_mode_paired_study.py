import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi import postcontact_paired_study as study


def declaration():
    return dict(
        postcontact_study_profile=study.LOCOMOTION_PROFILE,
        postcontact_study_groups=study.LOCOMOTION_GROUPS.copy(),
        groups=study.LOCOMOTION_GROUPS.copy(),
        max_attempts=34,
        max_generations=1,
        plateau_generations=1,
        deadline_unix=1792209600,
        neural_training_authorized=False,
        teacher_auto_admission=False,
        physical_payload_reader_mode="fixed_after_first_qualified",
        locomotion_baseline_report_path="relative-test-fixture.json",
        locomotion_baseline_report_hash="sha256:" + "a" * 64,
    )


def test_only_locomotion_mode_may_differ_not_control_parameters():
    config = declaration()
    study.require_declaration(config)
    a, b = [study.paired_candidate(config, arm) for arm in ("baseline", "probe")]
    assert a == b
    assert set(a) == {"parameters", "swing"}
    assert a["swing"] == "bounded_contact_diagnostic"
    assert a["parameters"]["forward_cap_m_s"] == 1.4
    assert a["parameters"]["closing_gain_s_inv"] == 1.2


@pytest.mark.parametrize(
    "key,value",
    [
        ("postcontact_probe", {}),
        ("closing_gain_probe_s_inv", 3.0),
        ("groups", [28, 44]),
        ("postcontact_study_groups", [28, 44]),
        ("max_attempts", 100),
        ("max_generations", 2),
        ("plateau_generations", 2),
        ("deadline_unix", float("inf")),
        ("deadline_unix", True),
        ("physical_payload_reader_mode", None),
        ("locomotion_baseline_report_path", None),
        ("neural_training_authorized", True),
        ("teacher_auto_admission", True),
    ],
)
def test_fixed_coverage_budget_reader_and_no_mixed_interventions(key, value):
    config = declaration()
    config[key] = value
    with pytest.raises(ValueError):
        study.require_declaration(config)


def test_baseline_parity_checks_every_array_and_dtype(tmp_path):
    a = dict(canonical_qpos=np.zeros((300, 1, 43)), other=np.zeros((300, 1, 2)))
    first, second = tmp_path / "old.npz", tmp_path / "new.npz"
    np.savez(first, **a)
    np.savez(second, **a)
    receipts = [dict(witnesses=dict(trace=dict(path=str(p)))) for p in (first, second)]
    study.verify_archived_locomotion_baseline(*receipts)
    for change in ("numeric", "dtype", "missing"):
        b = copy.deepcopy(a)
        if change == "numeric":
            b["other"][299, 0, 0] = 1.0
        elif change == "dtype":
            b["other"] = b["other"].astype(np.float32)
        else:
            del b["other"]
        np.savez(second, **b)
        with pytest.raises(ValueError, match="baseline physical arrays"):
            study.verify_archived_locomotion_baseline(*receipts)
