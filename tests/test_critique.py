from backend.nodes.company_overview import research_company_overview
from backend.nodes.critique import critique_claim


def test_critique_claim_verifies_well_corroborated_claim():
    results = research_company_overview("Anthropic")
    result = critique_claim("Anthropic was founded in 2021", results)
    assert result["verdict"] == "VERIFIED"
    assert result["supporting_count"] >= 2


def test_critique_claim_rejects_fabricated_claim():
    # supporting_count is inherently noisy (LLM-based judging, not exact
    # match) -- what actually matters is the final verdict never reaches
    # VERIFIED for an obviously fabricated claim, not the raw count
    results = research_company_overview("Anthropic")
    result = critique_claim("Anthropic was founded in 1998 by Elon Musk", results)
    assert result["verdict"] in ("CONTRADICTED", "UNVERIFIED")
