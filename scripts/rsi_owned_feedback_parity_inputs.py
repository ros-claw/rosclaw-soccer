"""Fixed six-case zero-action parity of optional owned continuation, no learning."""

from scripts.rsi_early_feedback_calibration import prepare as original_prepare
from scripts.rsi_early_feedback_calibration import require_scope as original_scope


def require_scope(cfg):
    if (
        cfg.get("profile") != "six-early-owned-continuation-parity-v1"
        or cfg.get("reuse_owned_updates_only_on_second_branch") is not True
    ):
        raise ValueError("declared paired deep-fork/owned-continuation zero diagnostic required")
    normalized = dict(cfg, profile="six-zero-early-feedback80ms-v1")
    normalized.pop("reuse_owned_updates_only_on_second_branch")
    original_scope(normalized)


def prepare(path):
    return original_prepare(path, scope_validator=require_scope)
