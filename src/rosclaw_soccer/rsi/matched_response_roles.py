"""Explicit ancestry/retention roles for consumed response diagnostics.

These roles do not establish sealed Fresh or global ancestry independence.
Historical retention is evaluation-only even when its ancestry also occurs in
local training. Callers authenticate the original declaration before use.
"""

from typing import Any


def admit_role(row: dict[str, Any], original_partition: str | None) -> dict[str, Any]:
    family = row.get("family")
    cluster = row.get("source_cluster")
    context = row.get("context_index")
    if type(cluster) is not int or type(context) is not int or context < 0:
        raise ValueError("explicit integer source ancestry and context required")
    if family == "old252":
        if not 0 <= cluster <= 2 or not context < 18:
            raise ValueError("unknown historical ancestry or context")
        expected = "CONSUMED_ORIGINAL_RETENTION" if context >= 12 else "CONSUMED_LOCAL_DEVELOPMENT"
        if original_partition != expected:
            raise ValueError("authenticated original historical partition required")
        split = "retention" if context >= 12 else ("train", "validation", "test")[cluster]
    elif family == "public40":
        if original_partition is not None or not 0 <= cluster < 10 or not context < 40:
            raise ValueError("unknown public consumed course or ancestry")
        split = "train" if cluster < 6 else "validation" if cluster < 8 else "test"
    else:
        raise ValueError("unknown factual response family")
    return {
        **row,
        "original_partition": original_partition,
        "legacy_split_NOT_authoritative": row.get("split"),
        "split": split,
        "gradient_eligible": split == "train",
        "sealed_fresh": False,
    }
