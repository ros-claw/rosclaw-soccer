# ROSClaw Soccer agent instructions

## Maintainer workflow

The maintainer requested direct main-branch integration on 2026-09-08.
For rosclaw-soccer, review and test changes, commit, and push main without
creating new PRs. Prefer fast-forward integration; do not force-push or
overwrite unrelated worktrees. Code integration never promotes an experimental
policy: retain SIM_ONLY boundaries and failed candidate evidence.

ROSClaw Soccer Academy is a downstream, simulation-first application of
ROSClaw Core. Keep generic Practice, Growth, Dream, Memory, promotion, safety,
and evidence contracts in `ros-claw/rosclaw`; keep football worlds, skills,
players, teams, exams, and media in this repository.

## Safety boundary

- The public academy is `SIM_ONLY`.
- Never open ROS, DDS, serial, CAN, or a vendor SDK from this repository.
- Never describe simulation success as real-robot authorization.
- Video is downstream of physics evidence and is never promotion truth.
- Raw datasets, checkpoints, trajectories, and full MP4 files stay outside Git.

## Local checks

```bash
python -m compileall -q src tests
ruff check .
ruff format --check .
mypy src
pytest -q
```

Set `ROSCLAW_SOCCER_EVIDENCE` to the external evidence root before building
media. Set `ROSCLAW_SOCCER_DATA` for external datasets. Do not hard-code local
absolute paths in source or committed manifests.

## Experimental evidence checks

- For generated or expanded Python experiment runners, lint the final expanded
  source for undefined names (`ruff check --isolated --select F821,F822`) before
  running physics. Checking only a wrapper does not check its execution namespace.
- Pass potentially negative scientific-notation numbers as `--key=value`.
- A regulation-sized goal does not prove regulation ball dimensions. Inspect
  the compiled ball's size and body mass with `physics.native_ball_dimensions`.
  Historical humanoid training fixtures may deliberately use other dimensions;
  preserve their evidence but never inherit adult-regulation publicity claims.
- Changes of physical dimensions define a new experiment protocol. Do not pick
  dimensions for favorable outcomes or silently rewrite legacy certificates.
- Before allocating workers for a new receiving course bank, validate every
  declared course with `training.receiving_course_preflight.preflight_receiving_courses`.
  A perturbation of a boundary course can leave the launch domain. Reject the
  entire declaration before execution; do not clip, replace, or silently drop
  invalid cases after observing other results. Paired rollouts/replays remain
  executions of their source course, not additional independent courses.
