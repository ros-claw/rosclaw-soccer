"""Measured-contact handoff for frame-zero evaluation, not teacher lead labels."""


def enter_measured_recovery(
    *, completed_controls: int, first_contact_control: int | None, reference_horizon: int
) -> bool:
    if (
        type(completed_controls) is not int
        or type(reference_horizon) is not int
        or not 1 <= completed_controls <= reference_horizon <= 300
    ):
        raise ValueError("completed bounded real reference controls required")
    if first_contact_control is not None and (
        type(first_contact_control) is not int
        or not 0 <= first_contact_control < completed_controls
    ):
        raise ValueError("contact must have already been measured, never a future label")
    return completed_controls == reference_horizon or (
        first_contact_control is not None and completed_controls >= first_contact_control + 15
    )
