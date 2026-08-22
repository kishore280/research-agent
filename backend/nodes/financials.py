from backend.search import search


def research_financials(company_name: str) -> list[dict]:
    query = f"{company_name} funding rounds valuation revenue"
    results = search(query)
    return results
