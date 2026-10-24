"""Seal and admit interrupted refinement prefixes; never rerun valid worlds."""

import argparse
import json
import subprocess
from pathlib import Path

from scripts.rsi_native_context_recovery_common import digest


def prefix(root):
    paths = sorted(root.glob("trial*.json"), key=lambda p: int(p.stem[5:]))
    if not paths or [p.name for p in paths] != [f"trial{i}.json" for i in range(len(paths))]:
        raise ValueError("nonempty contiguous completed trial prefix required")
    if len(paths) >= 16:
        raise ValueError("only incomplete declared studies may resume")
    rows = []
    pins = {}
    for i, p in enumerate(paths):
        if p.is_symlink():
            raise ValueError("ordinary immutable completed rows required")
        row = json.loads(p.read_text())
        if row["index"] != i or set(row["artifacts"]) != {"initial", "physical", "A", "B"}:
            raise ValueError("complete indexed primitive row required")
        for art in row["artifacts"].values():
            path = root / art["path"]
            if (
                path.is_symlink()
                or path.resolve().parent != (root / f"trial{i}").resolve()
                or digest(path) != art["hash"]
            ):
                raise ValueError("confined whole hashed primitive required")
            pins[str(path)] = art["hash"]
        pins[str(p)] = digest(p)
        rows.append(row)
    return rows, pins


def admit_recovery(path, expected_hash, state, model_hash, pins):
    if path.is_symlink() or digest(path) != expected_hash:
        raise ValueError("exact sealed interruption required")
    report = json.loads(path.read_text())
    if report["status"] != "FOUR_STORAGE_INTERRUPTED_PREFIXES_SEALED_NOT_PHYSICS_PROOF":
        raise ValueError("explicit failed-storage prefix seal required")
    item = next(x for x in report["shards"] if x["state_id"] == state)
    root = Path(item["root"])
    if root.is_symlink() or (root / "complete.json").exists():
        raise ValueError("original terminal partial study required")
    dp = root / "declaration.json"
    if digest(dp) != item["declaration_hash"]:
        raise ValueError("unchanged interrupted declaration required")
    d = json.loads(dp.read_text())
    if (
        d["source_head"] != report["producer_head"]
        or d["model_hash"] != model_hash
        or d["state_id"] != state
    ):
        raise ValueError("same original state/model declaration required")
    rows, raw_pins = prefix(root)
    if [digest(root / f"trial{i}.json") for i in range(len(rows))] != item["row_hashes"]:
        raise ValueError("entire sealed completed prefix required")
    for k, h in raw_pins.items():
        if k in pins and pins[k] != h:
            raise ValueError("conflicting partial ancestry")
        pins[k] = h
    pins[str(dp)] = item["declaration_hash"]
    pins[str(path)] = expected_hash
    return root, rows, item


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("producer-head", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    if a.output.exists() or a.root.is_symlink():
        raise ValueError("unused seal and canonical original data directory required")
    shards = []
    for state in (20, 23, 26, 29):
        root = a.root / f"state{state}" / "source"
        unit = f"rosclaw-two-body-response-state{state}-source-20261024.service"
        status = dict(
            line.split("=", 1)
            for line in subprocess.check_output(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
                    "-p",
                    "MainPID",
                    "-p",
                    "ActiveState",
                    "-p",
                    "Result",
                ],
                text=True,
            ).splitlines()
        )
        if status != dict(MainPID="0", ActiveState="failed", Result="exit-code"):
            raise ValueError("authoritatively terminal failed process required before sealing")
        dp = root / "declaration.json"
        d = json.loads(dp.read_text())
        if (
            d["source_head"] != a.producer_head
            or d["state_id"] != state
            or (root / "complete.json").exists()
        ):
            raise ValueError("same incomplete study required")
        rows, _ = prefix(root)
        dirs = sorted(int(x.name[5:]) for x in root.glob("trial*") if x.is_dir())
        if dirs != list(range(len(rows) + 1)):
            raise ValueError("one retained uncommitted interrupted world required")
        shards.append(
            dict(
                state_id=state,
                root=str(root),
                declaration_hash=digest(dp),
                row_hashes=[digest(root / f"trial{i}.json") for i in range(len(rows))],
                reusable_worlds=len(rows),
                uncommitted_worlds=1,
                uncommitted_native_steps_upper_bound=4000,
                exact_uncommitted_steps_known=False,
                terminal_service=status,
                unit=unit,
            )
        )
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="FOUR_STORAGE_INTERRUPTED_PREFIXES_SEALED_NOT_PHYSICS_PROOF",
            source_head=a.expected_head,
            producer_head=a.producer_head,
            shards=shards,
            storage_failure="ENOSPC /code; originals moved intact to canonical /home/dell evidence",
            reusable_worlds=sum(x["reusable_worlds"] for x in shards),
            uncommitted_worlds=4,
            uncommitted_native_steps_upper_bound=16000,
            physical_steps_this_seal=0,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
