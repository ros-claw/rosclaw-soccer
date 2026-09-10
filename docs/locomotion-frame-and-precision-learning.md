# Idle-frame selection and precision exploration

These are opt-in training/proposal primitives, not a new default gait or a
qualified football policy. They own no physics, recurrent-state reset, joint
command, admission, deployment, or hardware authority.

## Why a continuous command needs a discrete-frame contract

The experimental recurrent locomotion bridge selects a mirrored coordinate
frame whenever local lateral velocity is below `-1e-6 m/s`. Near a stopped ball,
small navigation exploration can repeatedly change that frame even though the
intended displacement is tiny. The existing learned motor also depends on its
original **unreflected idle convention**. Holding the last mirrored frame is
therefore not an equivalent fix.

S523 used 128 matched initial states per condition, shared standard-normal
draws, one frozen receiving actor, one live frozen LSTM step per control tick,
and unchanged navigation, joint, torque and pitch bounds. Results are isolated
single-body MuJoCo-Warp episodes, not 4v4 matches:

| Raw navigation noise σ | Frame rule | Continuous reception | Body-safe | Nonfoot contact |
| --- | --- | --- | --- | --- |
| 0 | Legacy | 56/128 | 128/128 | 1/128 |
| 0 | Contact-phase idle band | 56/128 | 128/128 | 1/128 |
| 0.005 | Legacy | 6/128 | 128/128 | 0/128 |
| 0.005 | Contact-phase idle band | 58/128 | 128/128 | 0/128 |
| exp(-2.5) ≈ 0.0821 | Legacy | 4/128 | 119/128 | 17/128 |
| exp(-2.5) ≈ 0.0821 | Contact-phase idle band | 10/128 | 123/128 | 18/128 |

The small-noise median mirror-switch count fell from 7 to 1. Large exploration
still fails often: a frame fix is not permission to ignore motion quality.
All conditions had **zero next-pass-pose qualification**. This experiment
isolates one obstacle to learning; it does not show a successful second pass.

Earlier alternatives were explicitly rejected. S521 symmetric hysteresis and
S522 fixing the entry frame for the entire skill both lost zero-noise receiving
success. Do not apply either globally based on the success of S523.

## Public API

`training.locomotion_frame` provides scalar and detached tensor versions:

```python
mode = select_locomotion_reflection(
    local_lateral_velocity_mps,
    prefer_idle_frame=measured_idle_compatible_phase,
)
```

The caller must determine phase applicability from current/past observations.
For the tested receiver only, this means **after its own measured foot contact,
while the receiving skill still owns the proposal interval**. No future outcome,
episode index, nominal contact time, or success label determines this flag.

With no declared idle phase, the legacy sign rule remains unchanged. Inside an
explicit idle phase and `abs(local lateral velocity) <= 0.02 m/s`, select the
model's declared idle frame, default `False`. Outside that band use the legacy
sign rule. Neither velocity nor recurrent state is modified. Models with other
idle conventions require separate validation; this is not a universal robot
balance controller.

The batched helper validates finite detached float32/64 velocities and aligned
boolean phase flags. Torch is imported lazily. The public helper reproduced all
76,800 archived S523 frame choices exactly. The zero-noise groups had identical
frame decisions and equal success counts, **not bit-identical physical arrays**;
the parity receipt preserves that distinction.

## PPO exploration floor

`FullBodyPPOUpdateConfig.minimum_log_std` defaults to the historical `-2.5` and
may explicitly be lowered to `-6.0`. The upper clamp remains `-0.3`. This changes
the exploration distribution, not the action/residual envelope.

Collectors and updates must use the same floor:

```python
config = FullBodyPPOUpdateConfig(
    observation_size=136,
    action_size=3,
    minimum_log_std=-5.5,
)
distribution = Normal(mean, agent.logstd.clamp(config.minimum_log_std, -0.3).exp())
```

Starting-policy likelihood and critic checks still run before any optimizer
step. Fine-noise data supplied with the default floor are rejected without
changing parameters, optimizer state, or RNG. No likelihood-tolerance relaxation
was introduced. A caller still owns rollback if an optimizer raises.

## Evidence and checks

Local evidence:

- `/home/dell/rosclaw_soccer_evidence/s523-contact-phase-frame-deadband-probe-v1/`
- `/code/rosclaw/phase8_evidence/s525-public-frame-and-precision-validation-v1/`

31 new tests cover scalar/batch frame semantics, malformed inputs, explicit
phase flags, precision likelihood rejection and exact optimizer replay. The
focused combined suite passed 86 tests; Ruff passed; mypy passed all 448 source
files; compileall passed. Full pytest: **1844 passed, 29 skipped, 11 failed**.
The 11 failure names exactly match the historical evidence-binding failures
recorded in S488; the repository is not claimed fully green.

The updated PPO implementation replayed the historical S517 first GPU update
with its default floor: maximum parameter error **0**. S526 integrated the
public frame helper into the actual eight-body receiving path, with no learned
navigation correction. All **185 numeric trajectory arrays, 4 motor-state
arrays and 12 locomotion-state arrays** were bit-identical to the prior
successful fixed-skill-0 case. Continuous reception still passed, all eight
bodies stayed safe and there were no robot collisions. The next outlet pass
still did not launch; this is an integration regression check, not a match win.

Fresh independent physics, the actual multi-body proposal path, continuous
control, next-skill readiness and subsequent passing remain separate acceptance
stages. Neither these utilities nor a successful training audit promotes a policy.
