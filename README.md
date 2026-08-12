# ⚽ ROSClaw Soccer Academy

**Growing self-improving humanoid footballers — from first steps to complete play.**

> See the world. See yourself. See others. Then grow.

ROSClaw Soccer Academy is the downstream flagship application for proving that
a physical agent can practice, fail, remember, dream, learn, compete, retain
old skills, and produce a verifiable growth history through ROSClaw.

Our first player is **Claw-7**, an original Unitree G1 footballer. Its long-term
training ethic is inspired by elite players such as Cristiano Ronaldo; it does
not reproduce a real person's likeness, identity, trademark, or footage.

## Current player — Academy Age 4

Academy Age is a sealed capability level, not wall-clock time or training
steps. Claw-7 is currently a provisional **Age 4: First Footballer**.

| Capability | Current evidence |
| --- | --- |
| Bilateral finishing | Physical right- and left-foot contact; 2.4 cm and 6.1 cm target error |
| Shared-world football | One ball, passer, shooter, and reactive goalkeeper in one MuJoCo world |
| Pass → finish | 2.90 m pass at 3.6 cm error; 6.61 m shot at 1.8 mm error |
| Recovery | No falls or joint-limit violations in the certified Age-4 cases |
| Long-range highlight | 7.5 m upper-corner shot, explicitly `DEVELOPMENT` due to saturation |

The weakness list remains visible: the approach-to-strike transition is not yet
fully unified, left-foot dynamic approach is immature, the goalkeeper reacts
but does not save the shot, and broad randomized-physics certification is still
missing.

**Age-4 homepage reel (certified cases):**
[watch/download 1080p](https://github.com/ros-claw/rosclaw-soccer/releases/download/age04-showcase-v0.1.0/claw7-academy-age04-hero-1080p.mp4)

**Separately trained regulation-physics candidate:**
[watch/download 1080p](https://github.com/ros-claw/rosclaw-soccer/releases/download/age04-showcase-v0.1.0/claw7-age04-regulation-development-1080p.mp4)
— 7.5 m shot, 3.5 mm goal-plane error, no fall/backward displacement. This
second video remains `DEVELOPMENT`, not certified, because 34 control steps
reached actuator saturation.

The latest local joint-growth curriculum now closes that candidate's missing
stability loop: a phase-conditioned support actor and a newly distilled
teacher-free contact actor pass all six Age-4 axes at 2.04 cm goal-plane error,
zero saturation, zero backward displacement, and strict replay. It remains a
`SIM_ONLY` development result until the harder true-frame-corner and randomized
retention exams pass. See the [v3 joint-growth report](docs/age04-joint-growth-v3.zh-CN.md).

## Regulation football physics

The simulator selects measurements inside the current IFAB ranges: a
105 × 68 m field, 7.32 × 2.44 m inside goal, 0.10 m frame and field lines,
and a 0.69 m circumference / 0.43 kg ball. Net depth and MuJoCo contact
coefficients are explicitly non-normative engineering parameters.

The Soccer Reality Pack now checks drop/compression, matched-spin rolling,
slide-to-roll transition, post rebound, and compliant-net retention before
training. Its five CPU MuJoCo cases pass with the current symmetric five-value
ball/pitch contact tuple. This fixes the legacy three-value contact tuple whose
second tangential coefficient was effectively near zero and could make the
ball look as if it slid like a cube.

## Attach to ROSClaw

Install this downstream package beside a ROSClaw checkout that includes the
`rosclaw.cli_extensions` interface:

```bash
python -m pip install --no-deps -e .
export ROSCLAW_SOCCER_EVIDENCE=/code/rosclaw/phase8_evidence
rosclaw soccer doctor
rosclaw soccer academy status
rosclaw soccer player show claw7
rosclaw soccer physics benchmark --output-dir /tmp/rosclaw-soccer-reality-pack
```

The package contributes four minimum-authority entry points:

| Entry point | Contribution | Authority ceiling |
| --- | --- | --- |
| `rosclaw.cli_extensions` | namespaced `rosclaw soccer` commands | no runtime/driver handle |
| `rosclaw.growth.adapters` | football experience normalization and diagnosis | evidence only |
| `rosclaw.simforge.tasks` | Age-4 and Age-5 task descriptions | discovery does not run physics |
| `rosclaw.dataset.sources` | relative-path-only motion-data labels | no root/file handle |

Installing the package adds these descriptions; uninstalling it leaves the
Core CLI, Growth, Dataset Doctor, and SimForge contracts usable. No extension
receives robot runtime, driver, actuator, ROS, or hardware authority.

## Evidence before highlights

The Age-4 reel is built only after source hashes and verdicts are checked:

```text
physics evidence → verdict → verified source media → reel → derivative manifest
```

Build the local 1080p H.264 reel (the MP4 stays outside Git):

```bash
export ROSCLAW_SOCCER_EVIDENCE=/code/rosclaw/phase8_evidence
python -m rosclaw_soccer.media.age04_reel \
  --manifest evidence/manifests/age04.json \
  --output /code/rosclaw/rosclaw_football/evidence/age04/claw7-academy-age04-hero-1080p.mp4
```

Every frame remains `SIM_ONLY`. Media is visualization, never promotion truth.

Every implementation milestone that changes observable football behaviour must
also publish a reviewable stage video outside the source checkout. The stage is
not complete until the video:

- replays the same immutable trajectory used by the JSON evidence;
- binds the evidence, trajectory, and renderer hashes in a sidecar manifest;
- passes post-encode resolution, frame-rate, frame-count, and duration checks;
- shows the complete approach, contact, goal-plane crossing, and recovery tail;
- labels rejected evidence as `DEVELOPMENT · NOT PROMOTED · SIM ONLY`.

The renderer is downstream of scoring: pixels never change the verdict. A
rejected candidate may be rendered only with an explicit review flag:

```bash
rosclaw soccer media free-kick \
  --evidence /outside/source/g1-free-kick.json \
  --asset-root /path/to/RoboNaldo_Deploy \
  --output /outside/source/stage-video.mp4 \
  --source-checkout /path/to/rosclaw-soccer \
  --resolution 1080p \
  --allow-rejected-candidate
```

Train a fresh regulation-physics Age-4 contact actor (eight bound probes,
teacher-free final replay):

```bash
rosclaw soccer academy train-age04 \
  --asset-root /path/to/RoboNaldo_Deploy \
  --gait-policy-root /path/to/g1/policy \
  --sonic-model-root /path/to/GEAR-SONIC \
  --seed-request /path/to/seed/request.json \
  --approach-strike-candidate /path/to/base-candidate.json \
  --source-checkout /path/to/rosclaw \
  --output-dir /outside/source/age04-regulation-training
```

`--football-motion-prior` is optional; a non-zero curriculum blend must be
retrained in the same eight-probe context. See the
[regulation Age-4 v2 report](docs/age04-regulation-training-v2.zh-CN.md) for the
former dynamic-stability boundary and the
[v3 joint-growth report](docs/age04-joint-growth-v3.zh-CN.md) for its closure.

## The growth loop

```text
Practice → Segment → Diagnose → Learn/Dream → Candidate
        → matched physics A/B → Retention → Promote or Reject
```

Generic Growth, Dream, Memory, Practice, safety, and promotion mechanisms stay
in [`ros-claw/rosclaw`](https://github.com/ros-claw/rosclaw). Football worlds,
skills, player cards, teams, leagues, exams, and media live here.

S1 extraction has moved the Age-4 approach/contact contracts, motion prior,
phase-conditioned residual, contact actor, and no-pickle G1 inference provider
behind this downstream boundary. Artifact schemas and hashes remain compatible
with the frozen pre-extraction implementation. See the
[S1 extraction report](docs/domain-extraction-s1.zh-CN.md).

S2 extraction now owns the regulation stadium, goal/net/ball physics, free-kick
runner, run-up providers, contact/recovery bridges, and football-specific expert
memory below the same boundary. A canonical strict replay produced a byte-identical
trajectory and zero result-field differences from the frozen pre-extraction run.
See the [S2 extraction report](docs/domain-extraction-s2.zh-CN.md).

## Next milestone

**Academy Age 5: First Touch** will require a continuous closed loop:

```text
receive rolling ball → control → remain ready within 1 s → pass or shoot
```

A single beautiful kick is not football. The episode must continue after the
ball leaves the foot.

## Safety

The public academy is `SIM_ONLY`. Simulation success is not real-robot
authorization. Real G1 work requires a separate ROSClaw hardware qualification
and operator-controlled safety path.
