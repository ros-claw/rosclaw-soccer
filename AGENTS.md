# ROSClaw Soccer agent instructions

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
