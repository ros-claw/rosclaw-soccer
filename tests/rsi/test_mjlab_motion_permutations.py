"""Parse literal data ABI without executing external package or deployment code."""

import pytest

from scripts.rsi_qualify_mjlab_motion_dataset import permutations


def test_complete_literal_orders(tmp_path):
    p = tmp_path / "loader.py"
    p.write_text(
        "_IL_TO_MJCF_JOINT="
        + repr(list(reversed(range(29))))
        + "\n_IL_TO_MJCF_BODY="
        + repr(list(range(30)))
    )
    j, b = permutations(p)
    assert j == list(reversed(range(29))) and b == list(range(30))


@pytest.mark.parametrize("kind", ["missing", "duplicate", "boolean", "repeated", "expression"])
def test_bad_or_executable_order_not_accepted(tmp_path, kind):
    j, b = list(range(29)), list(range(30))
    if kind == "duplicate":
        j[-1] = 0
    if kind == "boolean":
        j[0] = False
    text = "_IL_TO_MJCF_JOINT=" + repr(j) + "\n_IL_TO_MJCF_BODY=" + repr(b)
    if kind == "missing":
        text = text.split("\n")[0]
    if kind == "repeated":
        text += "\n_IL_TO_MJCF_JOINT=" + repr(j)
    if kind == "expression":
        text = "_IL_TO_MJCF_JOINT=list(range(29))\n_IL_TO_MJCF_BODY=" + repr(b)
    p = tmp_path / "loader.py"
    p.write_text(text)
    with pytest.raises(ValueError):
        permutations(p)
