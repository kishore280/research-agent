from backend.nodes.classify_focus_area import classify_focus_area


def test_classifies_financials_request():
    assert classify_focus_area("go deeper on their funding and revenue") == "financials"


def test_classifies_news_request():
    assert classify_focus_area("what's the latest news about them") == "news"


def test_classifies_industry_request():
    assert classify_focus_area("tell me more about their competitors and market") == "industry"


def test_classifies_overview_request():
    assert classify_focus_area("who founded the company and when") == "overview"
