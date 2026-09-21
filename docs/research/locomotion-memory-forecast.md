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
feedback control subsequently failed to rescue either course, as detailed below.

Other players, original solver warm-start, compliant nets and future tactical
decisions are not restored. Future residual changes are also omitted. E2 remains
unqualified; no neural weights have been updated or promoted by these audits.

## Actual feedback results: predictor accuracy was not sufficient

Four bounded, separately receipted comparisons each ran 12 physical executions
(two already-seen cases, three conditions, exact independent replay). These are
not 48 independent courses. All retained old-control physics and safe execution;
none rescued either receiving failure under the original examination.

| Evidence directory under `m0/` | Intervention | Safe rescues |
| --- | --- | --- |
| `recurrent-mpc-teacher-v1` | Recurrent foundation at the fixed 200 ms horizon | 0/2 |
| `clearance-mpc-teacher-v1` | Prefer predicted 5 mm shin clearance, unchanged ten candidates | 0/2 |
| `clearance-mpc-teacher-500ms-v1` | Qualified 500 ms forecast, same candidate family | 0/2 |
| `surface-mpc-teacher-500ms-v1` | Add two actual-contact-surface gradient candidates | 0/2 |

The first recurrent controller's first receiving windows failed specifically on
post-foot non-foot contact: two recorded right-shin contact frames in each case.
The force-bearing events were 33.525/1.024 N at 1.28/1.30 s and 26.657/1.999 N
at 1.32/1.34 s. Surface reconstruction authenticated the compiled full-world
model before interpreting contact IDs (`recurrent-mpc-contact-surfaces-v1.json`).
Two retained successful references kept roughly 7–9 mm minimum shin separation.
This is a diagnostic comparison, not a replacement success criterion.

The longer predictor first passed an independent 104-forecast failure-state
check (`failed-recurrent-contact-predictor-500ms-v1`): near-ball mean foot errors
fell from 0.023833/0.046157 m to 0.004150/0.002646 m. It still did not produce
successful control. By the relevant preparation frames the local candidate set
often contained no clearance-qualified option. Adding surface-aware candidates
also failed, despite analytical derivatives agreeing with finite differences to
about 8e-10 (`surface-contact-candidate-audit-v1`). The ankle's surface-distance
leverage was about 4–7 times that measured at its body origin. Correct geometry
alone was not enough to coordinate the full preparation and braking sequence.

The generic optional signed-distance Jacobian belongs in Core; Soccer owns the
foot/shin pair selection and research proposals. The Core surface extension is
covered by 23 surface tests and 345 local simulation tests. It exposes a local
linearization, not a clearance certificate or permission to execute motion.

Stop extending this constant-offset local candidate family. The next bounded
hypothesis is a time-varying prior assembled from authenticated existing
successful schedules, with target-role exclusion and explicit seen-data labels.
First qualify exact reduction to constant forecasts and original cursor filters;
then compare actual execution. This budget decision is not the Core blind-bank
plateau gate: no new sealed/blind bank has been consumed.

## Reuse and evidence loop

`providers/g1/locomotion_replica.py` packages the privately owned frozen predictor
controller. Asset, configuration and adapter hashes are required; restoring
memory and previous raw action precedes inference. Readouts are owned copies,
and invalid inference requires explicit restoration before reuse. The packaged
implementation reproduced 598 existing state/input/action/memory transitions
exactly (`packaged-locomotion-input-replay-v1`). This repeats existing references;
it adds no independent physical case or new neural training.

`practice-recurrent-mpc-memory-v1/complete.json` retains two authenticated success
fixtures and two recurrent-controller failures through Core record, strict
verification, distillation, local ingestion and query. Two successes and two
failures were retrieved; repeated failure ingestion stayed at two. Pre-inference
memory, measured physical outcomes and counterfactual forecasts remain separate.
This is reusable failure memory, not evidence that a student learned the skill.
