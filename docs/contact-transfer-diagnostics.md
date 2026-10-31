# Contact transfer diagnostics

`rosclaw_soccer.physics.contact_signature` compares the effective parameters
of an observed single contact in two MuJoCo models. It is a diagnostic for
investigating simulation transfer, not a promotion gate or a claim that two
models behave equivalently.

Call `mujoco.mj_forward(model, data)` or step each simulator first, then call
`observe_contact(model, data, geom_a, geom_b)` using IDs local to that model.
It requires exactly one active contact; missing, inactive and multiple-contact
pairs fail closed. `compare_contacts(source, target)` compares contact dimension,
the friction dimensions actually used, solver parameters and inclusion margin.
In particular, a `condim=3` contact does not use the stored rolling friction
coefficients; a `condim=6` contact does. The values are those reported by the
solver, including MuJoCo's active-friction minimum, not merely XML inputs.

Matching these fields is not enough to establish equivalent dynamics. Model
geometry, mass and inertia, solver configuration, initial state, surrounding
contacts, actuators and control policy also matter. Use paired trajectories,
physical task outcomes and independent validation before drawing conclusions.
This diagnostic never changes or authorizes robot motion.
