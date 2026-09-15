"""Select a residual learning slot without overriding motion or safety guards.

Selected roles may learn only inside an already admitted motor option. Other
roles retain the historical contact-teacher slot. This is SIM control routing,
not a capability grant, hardware Permit, or candidate promotion.
"""

from __future__ import annotations


def residual_skill_selected(
    *,
    role: str,
    option_only_roles: tuple[str, ...] | None,
    is_contact_teacher: bool,
    is_motor_option: bool,
) -> bool:
    if option_only_roles is not None and role in option_only_roles:
        return is_motor_option
    return is_contact_teacher and not is_motor_option


def residual_skill_preempted(
    *,
    role: str,
    option_only_roles: tuple[str, ...] | None,
    is_motor_option: bool,
) -> bool:
    """Clear residual memory on slot exit; do not leak it into another skill."""
    if option_only_roles is not None and role in option_only_roles:
        return not is_motor_option
    return is_motor_option
