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

## Attach to ROSClaw

Install this downstream package beside a ROSClaw checkout that includes the
`rosclaw.cli_extensions` interface:

```bash
python -m pip install --no-deps -e .
export ROSCLAW_SOCCER_EVIDENCE=/code/rosclaw/phase8_evidence
rosclaw soccer doctor
rosclaw soccer academy status
rosclaw soccer player show claw7
```

The adapter contributes only a namespaced command tree. It receives no robot
runtime, driver, actuator, ROS, or hardware authority.

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

## The growth loop

```text
Practice → Segment → Diagnose → Learn/Dream → Candidate
        → matched physics A/B → Retention → Promote or Reject
```

Generic Growth, Dream, Memory, Practice, safety, and promotion mechanisms stay
in [`ros-claw/rosclaw`](https://github.com/ros-claw/rosclaw). Football worlds,
skills, player cards, teams, leagues, exams, and media live here.

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
