"""Preflight all32 fixed contact perturbations; no policy inference or physics steps."""

import argparse
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for n in ("data", "pipeline"):
        p.add_argument("--" + n + "-root", type=Path, required=True)
        p.add_argument("--" + n + "-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and clean frozen neighborhood declaration required")
    from rosclaw_soccer.rsi.paid_neighborhood_sources import load_neighborhood_sources
    from rosclaw_soccer.rsi.paid_neighborhood_study import neighborhood_config

    data, cases, pins, specifications = load_neighborhood_sources(
        a.data_root, a.data_hash, a.pipeline_root, a.pipeline_hash
    )
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    try:
        import numpy as np

        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_skill_handoff import snapshot

        bench, _, scene = open_paid_bench(pins, cases)
        if scene != data["body_scene_hash"]:
            raise ValueError("same physical body/ball required")
        entries = []
        for case in cases:
            ep = bench.make(case)
            if ep.clip.length != 289 or ep.env.model.opt.timestep != 0.002:
                raise ValueError("fixed reference horizon and native clock required")
            q, v = ep.layout.canonical(ep.env.data)
            if (
                not np.array_equal(q[36:39], case["ball_world_initial"])
                or not np.array_equal(v[35:37], case["velocity_world_initial"])
                or not np.array_equal(ep.goal, case["goal_world"])
            ):
                raise ValueError("compiled fixture differs from complete declaration")
            path = a.output / f"case{case['case']}-initial.npz"
            np.savez_compressed(path, **snapshot(ep.env.data))
            entries.append(dict(case=case["case"], raw_path=path.name, raw_hash=digest(path)))
        if bench.native_steps or bench.neural_rows:
            raise ValueError("preflight must allocate neither rollouts nor neural inference")
        common = dict(
            source_head=a.expected_head,
            config=neighborhood_config(),
            cases=cases,
            specifications=specifications,
            body_scene_hash=scene,
            data_root=str(a.data_root),
            data_hash=a.data_hash,
            pipeline_root=str(a.pipeline_root),
            pipeline_hash=a.pipeline_hash,
            input_file_hashes=pins,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        write_once(a.output / "declaration.json", common)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_CONTACT_NEIGHBORHOOD32_PREREGISTERED_NO_ROLLOUT",
                **common,
                declaration_hash=digest(a.output / "declaration.json"),
                entries=entries,
                cases_preflighted=32,
                own_native_steps=0,
                own_neural_rows=0,
                Fresh_fit_rows=0,
                DEV_fit_rows=0,
                optimizer_updates=0,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
                own_native_steps=0,
                optimizer_updates=0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
