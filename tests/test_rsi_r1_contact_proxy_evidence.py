"""Fast contact-proxy evidence must fail closed on an unsealed motor tape."""

from pathlib import Path

import pytest
from rsi_r1_contact_proxy_fidelity_v139 import verify_one


def test_proxy_rejects_unsealed_motor_tape(tmp_path: Path):
    tape = tmp_path / "course.npz"
    tape.write_bytes(b"not a qualified eight-player tape")
    with pytest.raises(ValueError, match="sealed eight-player motor tape"):
        verify_one(tmp_path, tape, "sha256:" + "0" * 64)
