import json

import pytest

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_lateral_tracking_v286 import checked_execution_parent


@pytest.mark.parametrize("fault", (None, "source", "asset", "seal"))
def test_parent_provenance_is_checked_without_a_native_worker(tmp_path, fault):
    runner, asset, path = (tmp_path / n for n in ("runner.py", "body.usda", "report.json"))
    # Fixtures only. No simulator, body or actuator is instantiated.
    runner.write_text("simulation fixture")
    asset.write_text("body fixture")
    report = dict(
        source_hash=hash_bytes(runner.read_bytes()), asset_hash=hash_bytes(asset.read_bytes())
    )
    if fault in ("source", "asset"):
        report[f"{fault}_hash"] = "sha256:" + "a" * 64
    report["report_hash"] = hash_json(report)
    if fault == "seal":
        report["unsealed_edit"] = True
    path.write_text(json.dumps(report))
    if fault is None:
        assert checked_execution_parent(path, runner=runner, asset=asset) == report
    else:
        with pytest.raises(ValueError, match="before actor allocation"):
            checked_execution_parent(path, runner=runner, asset=asset)
