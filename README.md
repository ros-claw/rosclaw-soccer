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
| Shared-world football | One ball and three independent role agents; old highlight withdrawn after rolling audit |
| Pass → finish | Corrected-physics retraining in progress; old 3.6 cm / 1.8 mm result relied on sliding |
| Recovery | No falls or joint-limit violations in the certified Age-4 cases |
| Long-range highlight | 7.5 m upper-corner shot, explicitly `DEVELOPMENT` due to saturation |

The sealed Age-4 weakness list remains visible: the approach-to-strike
transition is not yet fully unified, left-foot dynamic approach is immature,
and broad randomized-physics certification is still missing. Later research
stages now contain real goalkeeper saves, but do not retroactively rewrite the
older Age-4 certificate.

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

## Current advanced team track

S112 has moved the S111 reproducibility rules into task-neutral ROSClaw Core
SimForge and then requalified the same continuous right-control chain: four
physical G1s, two balls, pass, first high strike, airborne save, measured
recovery/rearm, second-striker right-foot contact, second high glove save, and
final ready. Three fresh Python processes produced byte-identical trajectories;
Core independently recomputed seven process/content/authority gates while Soccer
retained only football physics scoring. The closure binds four source trees, 14
artifacts, numerical dependencies, and the process contract. This remains a
single-lane `SIM_ONLY` champion, not bilateral or hardware qualification. See the
[S112 Core-closure report](docs/s112-core-closure-adoption.zh-CN.md) and the
[historical S111 diagnosis](docs/s111-cross-process-current-runtime-requalification.zh-CN.md).

S113 starts the next role-isolated growth round without sacrificing that
champion.  The shared runtime now gives the second football independent
regulation mass/friction and passes the second striker's measured ball velocity
and goal target into its target-conditioned contact actor.  A plastic candidate
may replace the frozen parent only inside its declared launch envelope;
abstention falls back to the parent and acting outside support fails closed.  In
the current left-inner trial the parent retains the complete chain, while the
candidate supports and selects zero frames.  The 0.46 kg/high-grip holdout also
misses the second save.  The candidate is therefore explicitly rejected rather
than credited with the parent's success.  See the
[S113 stability–plasticity report](docs/s113-role-isolated-stability-plasticity.zh-CN.md).

S114 closes the first real failure-to-growth loop on that role boundary.  A
role-local teacher now projects forces through the second striker's own 29 DoFs
and rehearses eight safe/failed contact conditions inside the uninterrupted
four-G1 world.  The first distilled candidate genuinely selected itself but
still let the ball cross the goal line.  That bound failure then contracted the
inverse-model trust region and added proprioceptive foot-speed feedback.  The
updated candidate—not its frozen parent—passes two exact control replays with a
1.475 m physical glove save and final ready state.  It still fails the sealed
0.46 kg/high-grip holdout, so the portfolio gate rejects broad promotion and
turns that miss into the next curriculum.  See the
[S114 failure-memory report](docs/s114-failure-memory-contact-growth.zh-CN.md).

S115 turns the failed heavy-ball holdout into an explicit whole-body motion
curriculum instead of hiding it behind a larger foot impulse.  The sealed
failure-updated contact actor now runs with independently bound ball physics and
second-striker foot-pitch context.  At 0.46 kg, friction 0.16, and 0.1261 rad,
two strict replays produce a real 1.373 m glove save and a fully ready final
state.  A tiny sealed neighbour (0.12605 rad) still saves the ball but fails the
post-save readiness gate, so the portfolio correctly rejects broad promotion.
This exposes a narrow, non-smooth contact/recovery boundary for the next robust
policy-learning round.  See the
[S115 heavy-ball curriculum report](docs/s115-heavy-ball-motion-curriculum.zh-CN.md).

S116 converts that neighbour failure into a proprioceptive impact-recovery
reflex.  After real glove contact, the keeper may enter lower-body landing
capture only inside a measured time, bilateral-support, and root-speed
envelope; the existing learned recovery athlete remains responsible outside
that envelope.  The former `0.12605 rad` failure now repeats the physical save
and passes all eight final-ready gates, while the `0.1261 rad` control remains
unchanged.  A harder `0.12615 rad` neighbour still fails, so this is a local
two-context qualification rather than broad robustness or a new end-to-end
neural cerebellum.  See the
[S116 impact-recovery report](docs/s116-proprioceptive-impact-recovery.zh-CN.md).

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

S4a additionally closes the integration gap between that isolated Reality Pack
and actual three-player evidence. Translational and angular free-ball damping
are now dimensionally separate, and every passing trajectory must prove rolling
from measured linear and angular velocity. The frozen S3 relay is correctly
rejected as sliding; see the [S4a rolling report](docs/rolling-authenticity-s4a.zh-CN.md).

S4b now gives passer, shooter, and goalkeeper independent policy identity,
counterfactual contribution rewards, and a joint all-role promotion gate.  The
corrected-world drift diagnosis and first safe learning sweep are documented in
the [S4b multi-agent report](docs/multi-agent-growth-s4b.zh-CN.md).

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

S3 has extracted the evidence validator and evidence-downstream renderer for one
passer, one shooter, one reactive goalkeeper, and one shared ball. The first local
baseline binds all source and renderer hashes and visibly identifies its legacy
3.00 × 2.00 m training goal; it is the migration baseline for the next regulation-
goal rerun, not a regulation claim. See the
[S3 three-player report](docs/three-player-media-s3.zh-CN.md).

Build the three-player review video from an external frozen evidence bundle:

```bash
rosclaw soccer media three-player \
  --evidence /outside/source/g1-three-player-showcase.json \
  --asset-root /path/to/RoboNaldo_Deploy \
  --output /outside/source/three-player-stage.mp4 \
  --source-checkout /path/to/rosclaw-soccer \
  --resolution 1080p
```

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
