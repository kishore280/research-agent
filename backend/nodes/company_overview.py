from backend.search import search


def research_company_overview(company_name: str) -> list[dict]:
    query = f"{company_name} company overview founding history"
    results = search(query)
    return results
