from __future__ import annotations

import inspect

from rosclaw_soccer.media.neural_contact_canary_video import (
    render_neural_contact_canary_video,
    validate_neural_contact_canary_video,
)


def test_neural_contact_video_is_evidence_downstream() -> None:
    parameters = inspect.signature(render_neural_contact_canary_video).parameters
    for name in ("canary_report_path", "asset_root", "output_path", "source_checkout"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert callable(validate_neural_contact_canary_video)
