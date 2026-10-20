"""Six unchanged certified right anchors and three left-foot reference options."""


def build_candidates(predict, rows, prefix, source, item, folder):
    from scripts.rsi_response_cem_builder import certified_anchors

    certified_anchors(predict, rows, source, item)
    for law in (0, 3, 4):
        predict(
            law,
            (0.0, 0.0, 0.0),
            dict(method="sagittal_reference_and_reflected_entry_prior", intended_foot="left"),
            contact_reference_foot="left",
        )
    return [dict(law_index=law, intended_reference_foot="left") for law in (0, 3, 4)]
