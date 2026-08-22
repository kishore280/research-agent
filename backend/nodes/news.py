from backend.search import search


def research_news(company_name: str) -> list[dict]:
    query = f"{company_name} recent news announcements"
    results = search(query)
    return results
