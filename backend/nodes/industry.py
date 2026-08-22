from backend.search import search


def research_industry(company_name: str) -> list[dict]:
    query = f"{company_name} industry market position competitors"
    results = search(query)
    return results
