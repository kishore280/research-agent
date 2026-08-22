from typing import TypedDict

from langgraph.graph import END, StateGraph

from backend.nodes.critique import critique_claim
from backend.search import search

MAX_ATTEMPTS = 3


class ClaimState(TypedDict):
    claim: str
    sources: list[dict]
    verdict: str
    supporting_count: int
    attempts: int


def critique_node(state: ClaimState) -> dict:
    critique_result = critique_claim(state["claim"], state["sources"])
    
    return {
        "verdict": critique_result["verdict"],
        "supporting_count": critique_result["supporting_count"],
    }


def refine_search_node(state: ClaimState) -> dict:
    query = f"{state['claim']} fact check verify"
    new_results = search(query)
    return {
        "sources": state["sources"] + new_results,
        "attempts": state["attempts"] + 1,
    }


def should_retry(state: ClaimState) -> str:
    if state["verdict"] in ["VERIFIED", "CONTRADICTED"]:
        return "end"
    elif state["attempts"] < MAX_ATTEMPTS:
        return "retry"
    else:
        return "end"


def build_claim_graph():
    graph = StateGraph(ClaimState)
    graph.add_node("critique", critique_node)
    graph.add_node("refine_search", refine_search_node)
    graph.set_entry_point("critique")
    graph.add_conditional_edges("critique", should_retry, {"retry": "refine_search", "end": END})
    graph.add_edge("refine_search", "critique")
    return graph.compile()