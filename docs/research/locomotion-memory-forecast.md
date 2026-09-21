# Locomotion memory and receiving forecasts

Research-only, SIM_ONLY. This component qualification is not E1/E2, a trained
receiver, a complete simulator checkpoint, or a Champion promotion.

## Why this intervention

The qualified RoboNaldo TorchScript contains persistent LSTM `hidden_state` and
`cell_state`, each CPU float32 `(1, 1, 256)`, as well as a previous 29D raw action
in its next observation. The earlier private MuJoCo predictor held a joint target
for 200 ms. It did not evolve the frozen recurrent foundation. Several failed
MPC pilots therefore motivated checking controller-state reconstruction before
increasing candidate count, changing rewards, or training a student.

## Boundaries

`providers/g1/locomotion_memory.py` copies the two tensors into immutable byte
values bound to a canonical policy artifact hash. Capture neither forwards nor
detaches the live policy. Restore requires an explicitly private model, validates
both states before mutation, and rejects non-finite, aliased or incompatible
state layouts. This trusted ownership assertion is not an OS security sandbox.

The opt-in receiving recorder preserves the actual 96D input, raw action,
post-inference memory, body-order foundation target, world navigation command,
reflection flag, and same-clock pre-integration physical state. It does not
change controller configuration, the actor, filters, torque limits, or physics.
Default-off execution retains the previous trace and behavior.

An opt-in `ReceivingLocomotionContext` exposes immutable values to a feedback
proposal. Memory-requiring providers must request explicit recording; missing or
stale context and changed requirements are rejected. The original oracle cursor
still owns filtering, limits, admission and execution. No writable policy or
MuJoCo handle enters the proposal interface. Existing providers retain their
observation hashes when no recurrent context is requested.

## Evidence so far

External paths below are relative to `receiving-mechanism-reboot`.

- `m0/locomotion-memory-replay-v1`: integration failure retained. One successful
  baseline completed; the first recording aborted because the new contract
  expected a bare hash while Soccer uses `sha256:`. A regression test now binds
  the actual Soccer hash helper. No failed partial run was overwritten.
- `m0/locomotion-memory-replay-v2`: six executions, two already-seen successful
  four-knot oracle courses, baseline/record/replay. Both retained safe capture;
  old physical arrays and all baseline trace fields were exact. Each recorded
  execution independently reproduced 299 subsequent actions and LSTM states.
- `m0/locomotion-memory-replay-v3`: four executions adding same-clock qpos/qvel.
  Existing v2 fields and independent replay remained exact; no new skill credit.
- `m0/private-locomotion-input-replay-v1`: 598 recorded-state reconstructions.
  Rebuilding proprioception, normalized command, reflection and previous-action
  input exactly reproduced network inputs, raw outputs, body targets and memory.
  Future recorded values were used only for this offline reconstruction audit.
- `m0/recurrent-contact-predictor-v1`: 108 private forecasts, not 108 matches.
  At the same fixed 200 ms horizon, the predictor holds the current world command
  and residual while advancing the frozen LSTM every 20 ms using predicted body
  state. It never reads future commands. Near-ball mean foot error in the two
  seen positive cases fell from 0.040910/0.041525 m to 0.002110/0.004171 m.
  Ball errors changed from 0.000084/0.000106 m to 0.000166/0.000214 m: slightly
  worse, within the predeclared +0.01 m guard. Both cases passed the predeclared
  25% foot-error improvement and above-ground ankle-origin checks. Default-path
  reduction and private recurrent replay were exact.

These two cases were selected from existing known successes, not a blind bank.
The fixed failure-state check then used `m0/failed-locomotion-memory-v1`: four
executions of the two existing teacher-suppressed failures, both still safe
failures, all old trace fields exact, independent recording replay exact. The
corresponding `m0/failed-locomotion-input-replay-v1` reconstructed 598 further
inputs/outputs/targets/memory states exactly.

`m0/failed-recurrent-contact-predictor-v1` performed another 108 private forecasts.
Near-ball foot error fell from 0.009153/0.018913 m to 0.000387/0.000364 m; mean
ball errors improved too. Both cases passed the same predeclared predictor gate.
These are repeated known cases, not independent blind samples. Matched actual
feedback control is the next gate. More accurate forecasts have not yet rescued
a receiving course.

Other players, original solver warm-start, compliant nets and future tactical
decisions are not restored. Future residual changes are also omitted. E2 remains
unqualified; no neural weights have been updated or promoted by these audits.
