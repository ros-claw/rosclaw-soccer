from collections import Counter

from scripts.rsi_mesh_contact_catalog_reachability import declared_jobs


def test_all_existing_options_cover_both_original_hard_draws():
    jobs = declared_jobs()
    assert len(jobs) == 16
    assert len({j["job"] for j in jobs}) == 16
    assert {(j["group"], j["option_index"]) for j in jobs} == {
        (group, option) for group in (8, 56) for option in range(8)
    }
    assert Counter(j["group"] for j in jobs) == {8: 8, 56: 8}


def test_none_controls_first_and_no_result_dependent_schedule():
    assert declared_jobs()[:2] == [
        dict(job="option-0-raw-8", option_index=0, group=8),
        dict(job="option-0-raw-56", option_index=0, group=56),
    ]
    altered = declared_jobs()
    altered[0]["group"] = 999
    assert declared_jobs()[0]["group"] == 8
