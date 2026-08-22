from backend.graph import _is_duplicate_claim, route_research


def test_duplicate_claim_detected_when_reworded():
    existing = ["Anthropic was founded in 2021 by Dario Amodei and Daniela Amodei."]
    assert _is_duplicate_claim("Anthropic was founded in 2021.", existing)


def test_reworded_paraphrase_still_detected():
    # this is the case lexical word-overlap missed: same fact, mostly
    # different wording. Measured at 0.638 cosine similarity -- above
    # the calibrated 0.6 threshold but below the literature's cited 0.75,
    # which is exactly why that threshold was recalibrated (see llm.txt)
    existing = ["Anthropic was founded in 2021 by Dario Amodei and Daniela Amodei."]
    assert _is_duplicate_claim("Anthropic came into existence in 2021, started by two siblings.", existing)


def test_distinct_claim_not_flagged_duplicate():
    existing = ["Anthropic was founded in 2021 by Dario Amodei and Daniela Amodei."]
    assert not _is_duplicate_claim("Anthropic raised $4 billion from Amazon.", existing)
    assert not _is_duplicate_claim("Anthropic is headquartered in San Francisco.", existing)


def test_no_existing_claims_never_duplicate():
    assert not _is_duplicate_claim("Anthropic was founded in 2021.", [])


def test_route_research_full_run_targets_all_four():
    state = {"focus_area": None}
    routed = route_research(state)
    assert set(routed) == {"research_overview", "research_industry", "research_financials", "research_news"}


def test_route_research_followup_targets_only_focus_area():
    state = {"focus_area": "financials"}
    assert route_research(state) == ["research_financials"]
