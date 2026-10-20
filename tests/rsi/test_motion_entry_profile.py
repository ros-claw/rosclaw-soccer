"""Binding/geometry fixtures only; no dynamic admission or motion authority."""

from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
from rosclaw_soccer.rsi.motion_entry_profile import ConfiguredEntryEstimate, build_motion_entry


def fixture(side="right", window=(466, 767), transport=True):
    source = np.zeros((800, 36))
    source[:, 2], source[:, 3] = 0.79, 1.0
    source[466:, :2] = [-0.3, 0.2]
    prior = SkillEntryAffordance((0.2, 0.03), 150, 1.0, "sha256:" + "1" * 64)
    kwargs = dict(
        world_hash="sha256:" + "2" * 64,
        motion_artifact_hash="sha256:" + "3" * 64,
        reference_side=side,
        source_window=window,
        transport=transport,
    )
    b, e, p = build_motion_entry(source, prior, **kwargs)
    return b, e, p, prior, source, kwargs


def test_baseline_prior_identity_and_readonly_source():
    b, e, p, prior, source, kw = fixture(window=(286, 767), transport=False)
    assert e is prior and e.contract_hash == prior.contract_hash
    np.testing.assert_array_equal(p, source)
    assert not p.flags.writeable
    b.validate(p, e, world_hash=kw["world_hash"], source_window=(286, 767), reference_side="right")


def test_transported_kind_is_not_a_fabricated_historical_contact():
    b, e, p, prior, _, _ = fixture()
    assert isinstance(e, ConfiguredEntryEstimate)
    assert e.parent_affordance_hash == prior.contract_hash
    assert e.historical_frame == prior.historical_frame
    assert e.motion_configuration_hash == b.configuration.contract_hash
    np.testing.assert_allclose(e.ball_offset_xy_m, [0.5, -0.17])
    assert not b.report()["dynamic_feasibility_proven"]
    assert not b.report()["contact_qualified"]
    with pytest.raises(ValueError, match="original right prior"):
        build_motion_entry(
            p,
            e,
            world_hash=b.configuration.world_hash,
            motion_artifact_hash=b.configuration.motion_artifact_hash,
            reference_side="left",
            source_window=(466, 767),
            transport=True,
        )


def test_left_is_explicit_reference_augmentation_not_an_executed_foot_label():
    b, e, _, _, _, _ = fixture(side="left")
    np.testing.assert_allclose(e.ball_offset_xy_m, [0.5, 0.17])
    assert b.report()["reference_side_not_observed_foot"]


@pytest.mark.parametrize(
    "kind", ["source", "source_shape", "world", "window", "side", "entry", "config"]
)
def test_mismatches_rejected_before_body_creation(kind):
    b, e, p, _, _, kw = fixture()
    args = dict(world_hash=kw["world_hash"], source_window=(466, 767), reference_side="right")
    if kind == "source":
        p = p.copy()
        p[500, 0] = 1.0
    elif kind == "source_shape":
        p = p.reshape(-1)
    elif kind == "world":
        args["world_hash"] = "sha256:" + "4" * 64
    elif kind == "window":
        args["source_window"] = (286, 767)
    elif kind == "side":
        args["reference_side"] = "left"
    elif kind == "entry":
        e = replace(e, ball_offset_xy_m=(0.3, 0.1))
    else:
        altered = replace(b.configuration, world_hash="sha256:" + "4" * 64)
        b = replace(b, configuration=altered)
        args["world_hash"] = altered.world_hash
    with pytest.raises(ValueError):
        b.validate(p, e, **args)
