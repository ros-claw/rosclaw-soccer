from pathlib import Path
from types import SimpleNamespace

import numpy as np

import rosclaw_soccer.rsi.reference_tracking_episode as module
from rosclaw_soccer.providers.g1.sonic_pose_reference import SonicPoseReference


def test_each_frame_consumes_own_new_measurement_and_preserves_gain_contract(monkeypatch):
    defaults = module.G1SonicRunupController.default_angles
    entry = np.r_[0, 0, 0.793, 1, 0, 0, 0, defaults]
    reference = SonicPoseReference((tuple(float(x) for x in entry),) * 10, "sha256:" + "1" * 64)
    seen = []

    class Navigation:
        def __init__(self, root, agent, config):
            assert config.pose_reference is reference
            self.contract_hash = "fake-navigation-not-neural-evidence"
            self.backend = SimpleNamespace(
                kp=np.ones(29),
                kd=np.ones(29),
                planner_calls=0,
                qualification=SimpleNamespace(qualification_hash="fake-foundation"),
                action=np.zeros(29),
            )

        def propose(self, observation):
            seen.append(observation)
            assert observation.qpos[7] == defaults[0] + observation.frame / 1000
            assert observation.time_sec == observation.frame * 0.02
            return SimpleNamespace(target_rad=defaults, kp=np.ones(29), kd=np.ones(29))

    class Body:
        def __init__(self, *args, **kwargs):
            self.frame = 0
            self.kp, self.kd = kwargs["kp"], kwargs["kd"]
            self.data = SimpleNamespace(time=0.0)

        def canonical(self):
            q = np.r_[entry, 10, 0, 0.13, 1, 0, 0, 0].astype(float)
            q[7] += self.frame / 1000
            return q, np.zeros(41)

        def step(self, frame, target):
            assert frame == self.frame and target.shape == (29,) and target.dtype == np.float64
            q, _ = self.canonical()
            self.frame += 1
            self.data.time = self.frame * 0.02
            return dict(canonical_qpos=q[None], pelvis_z_per_substep_m=np.full((1, 10), 0.793))

        def review(self):
            assert self.frame == 300
            return dict(integrated_steps=3000, policy_causality_verified=False)

    monkeypatch.setattr(module, "G1SonicNavigation", Navigation)
    monkeypatch.setattr(module, "G1TargetEpisode", Body)
    monkeypatch.setattr(module.mujoco.MjModel, "from_binary_path", lambda path: object())
    arrays, report = module.run_reference_tracking_episode(
        Path("unused"), Path("unused.mjb"), world_hash="sha256:" + "2" * 64, reference=reference
    )
    assert len(seen) == 300
    assert arrays["canonical_qpos"].shape == (300, 43)
    assert arrays["neural_action_isaac"].shape == (300, 29)
    assert report["optimizer_steps"] == 0
    assert report["promotion_authorized"] is False
    assert report["body"]["policy_causality_verified"] is False
