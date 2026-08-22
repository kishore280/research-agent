from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from backend.claim_loop import build_claim_graph
from backend.nodes.company_overview import research_company_overview
from backend.nodes.decompose import decompose_into_claims
from backend.nodes.financials import research_financials
from backend.nodes.industry import research_industry
from backend.nodes.news import research_news

# capped for demo/testing
MAX_CLAIMS = 5


class ResearchState(TypedDict):
    company_name: str
    overview: list[dict]
    industry: list[dict]
    financials: list[dict]
    news: list[dict]
    all_findings: list[dict]
    claims: list[str]
    claim_results: list[dict]
    report: dict


def research_overview_node(state: ResearchState) -> dict:
    return {"overview": research_company_overview(state["company_name"])}


def research_industry_node(state: ResearchState) -> dict:
    return {"industry": research_industry(state["company_name"])}


def research_financials_node(state: ResearchState) -> dict:
    return {"financials": research_financials(state["company_name"])}


def research_news_node(state: ResearchState) -> dict:
    return {"news": research_news(state["company_name"])}


def collect_node(state: ResearchState) -> dict:
    all_findings = state["overview"] + state["industry"] + state["financials"] + state["news"]
    return {"all_findings": all_findings}


def decompose_node(state: ResearchState) -> dict:
    claims = decompose_into_claims(state["all_findings"])
    if len(claims) > MAX_CLAIMS:
        print(f"[decompose] {len(claims)} claims found, capping to {MAX_CLAIMS} for this run")
    return {"claims": claims[:MAX_CLAIMS]}


def critique_claims_node(state: ResearchState) -> dict:
    claim_graph = build_claim_graph()
    claim_results = []
    for claim in state["claims"]:
        result = claim_graph.invoke({
            "claim": claim,
            "sources": state["all_findings"],
            "verdict": "",
            "supporting_count": 0,
            "attempts": 0,
        })
        claim_results.append(result)
    return {"claim_results": claim_results}


def report_node(state: ResearchState) -> dict:
    report = {
        "company": state["company_name"],
        "claims": [
            {
                "claim": r["claim"],
                "verdict": r["verdict"],
                "supporting_count": r["supporting_count"],
                "attempts": r["attempts"],
            }
            for r in state["claim_results"]
        ],
    }
    return {"report": report}


def build_research_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("research_overview", research_overview_node)
    graph.add_node("research_industry", research_industry_node)
    graph.add_node("research_financials", research_financials_node)
    graph.add_node("research_news", research_news_node)
    graph.add_node("collect", collect_node)
    graph.add_node("decompose", decompose_node)
    graph.add_node("critique_claims", critique_claims_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "research_overview")
    graph.add_edge(START, "research_industry")
    graph.add_edge(START, "research_financials")
    graph.add_edge(START, "research_news")

    graph.add_edge("research_overview", "collect")
    graph.add_edge("research_industry", "collect")
    graph.add_edge("research_financials", "collect")
    graph.add_edge("research_news", "collect")

    graph.add_edge("collect", "decompose")
    graph.add_edge("decompose", "critique_claims")
    graph.add_edge("critique_claims", "report")
    graph.add_edge("report", END)

    return graph.compile()
