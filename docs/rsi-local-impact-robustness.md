# Local impact robustness pilot (SIM_ONLY)

The six closed C3 development controls improved RSI-M0 first-touch outcomes
from 4/6 to 6/6 using prospective bounded torque correction. This is not yet
a learned skill, general teacher, receiving/possession skill or team match.

`scripts/rsi_local_impact_robustness.py` is the next, explicitly **local** test.
It must not execute until the complete original motor/physics audit and the
selected-action legacy/owned-continuation parity have both terminated
successfully and their complete source/input chains authenticate.

## Question and falsification

Does the same current-body controller improve nearby ball conditions without
destroying original-control successes or safety? New unsafe outcomes veto a
local net-gain result. No net rescue, a genuine controller mismatch, or repeated
failure to reproduce improvements is evidence against this local mechanism,
not permission to loosen the physical task or train from failed teachers.

Even a local pass does not authorize training. The three source-course
clusters are all high-x balls rolling away. Wider public coverage, retention,
independent physical readback, and the subsequent three-seed learning study
remain separate obligations under the 1013 plan. Fresh remains sealed.

## Fixed design

For each of the three original courses, use both original heading bindings
and original policy seeds. Before any model allocation, declare both offsets
`(dx, dy, dvx)`:

- `(+0.01m, -0.015m, +0.02m/s)`;
- `(-0.01m, +0.015m, -0.02m/s)`.

Reject the whole declaration if any course leaves the existing public training
domain. Never clip, replace or silently drop a generated case. This produces
6 new physical courses, 12 policy/course paired contexts and **24 executions**,
not 24 independent scenarios. Reuse the existing 12 source baseline/correction
outcomes instead of rerunning their exact bodies for more sample count.

Each new context executes original control, then the same original policy with
prospective early impact enabled. Both arms recompute from their own current
body. The original-controller trace is a post-execution comparison oracle only;
it is not passed into the correction actor. Verify the complete original prefix
up to the first actual correction, including the current neural/body state.

The world, regulation ball, PD limits, H300, post-contact60 and M0 scoring stay
unchanged. Record all successful, failed, right-censored and unsafe outcomes.
There are no optimizer updates. Preserve each arm's trace, model identity,
preaction predictions/selection, physical counters and outcome. A fatal failure
preserves completed results, the partial trace and current body state; it does
not silently repeat the experiment.

## Execution contract

Use an immutable checkout and a new external output directory. Bind all
predecessor file hashes in a `local-early-impact-perturbation24-v1` config.
The existing restoration/native/H300/readback/recorded-physics fields are
required, together with `motor_unit`, `motor_root`,
`motor_complete_file_hash`, `owned_unit`, `owned_root` and
`owned_complete_file_hash`.

```bash
python scripts/rsi_local_impact_robustness.py --config /external/config.json --preflight
python scripts/rsi_local_impact_robustness.py --config /external/config.json
```

The whole-study bounds are 72,000 native steps plus at most 21,600 forecast
steps, 24 new episodes, 300 frames per episode, zero optimizer updates, an
explicit wall deadline and a 56GiB available-memory construction floor.
Do not run another full-model constructor concurrently. The runner is
single-process; no multiprocessing bootstrap or real robot transport is used.

Owned continuation is an explicit opt-in for this declared simulation study.
The public `CausalImpactReplay` and `EarlyImpactController` options remain
**False by default**. The option does not itself prove parity or confer
teacher/promotion authority.

The complete report includes every pair, primitive rescue/regression/unsafe
counts and source identities. Software tests, successful process exit and
video rendering are not substitutes for those physical results.
