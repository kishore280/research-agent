from backend.nodes.company_overview import research_company_overview


def test_research_company_overview_returns_relevant_results():
    results = research_company_overview("Anthropic")

    assert len(results) > 0

    for r in results:
        assert "title" in r
        assert "url" in r
        assert "content" in r

    combined_content = " ".join(r["content"].lower() for r in results)
    assert "2021" in combined_content or "amodei" in combined_content or "ai safety" in combined_content
