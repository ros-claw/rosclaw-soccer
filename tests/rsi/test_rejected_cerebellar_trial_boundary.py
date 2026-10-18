import pytest

from scripts.rsi_read_rejected_cerebellar_trials import rejection_line


def test_rejection_boundary_is_an_if_not_valid_not_a_string():
    assert (
        rejection_line(
            'def fit_update():\n    text = "if not valid:"\n    if not valid:\n        pass\n'
        )
        == 3
    )


@pytest.mark.parametrize(
    "body", ["pass", "if not valid:\n        pass\n    if not valid:\n        pass"]
)
def test_ambiguous_or_absent_boundaries_rejected(body):
    with pytest.raises(ValueError, match="one unmodified"):
        rejection_line("def fit_update():\n    " + body + "\n")
