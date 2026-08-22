import operator
from typing import Annotated, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from sentence_transformers import SentenceTransformer

from backend.claim_loop import build_claim_graph
from backend.nodes.company_overview import research_company_overview
from backend.nodes.decompose import decompose_into_claims
from backend.nodes.financials import research_financials
from backend.nodes.industry import research_industry
from backend.nodes.news import research_news

# capped for demo/testing
MAX_CLAIMS = 5

# compiled once at module load and reused across every critique_one_claim_node
# call -- rebuilding/recompiling a StateGraph per node execution is a known
# LangGraph inefficiency (community discussion + a reported memory-growth
# bug from doing this per-request), not something the docs recommend
_claim_graph = build_claim_graph()

# Sentence-BERT (Reimers & Gurevych 2019, https://arxiv.org/abs/1908.10084),
# same model project 1 uses for dense retrieval. Used here to catch a newly
# decomposed claim that just reworded one already in the report, so the
# critique loop doesn't re-verify it. First version used lexical word
# overlap -- switched after checking that claim-deduplication research
# reports embedding cosine similarity beating token-overlap methods at
# catching paraphrases; see Checkpoint 6 notes in llm.txt for the full
# reasoning and the honest limitation this doesn't fix.
# claim-dedup literature suggests ~0.75 cosine similarity as a starting
# threshold, but that doesn't transfer to this smaller model on short,
# single-sentence claims: empirical calibration against hand-crafted
# duplicate/non-duplicate pairs (see llm.txt Checkpoint 6 notes) found
# duplicates scoring as low as 0.638 and distinct claims scoring as high
# as 0.565 -- 0.6 is the threshold that actually separates them here
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
DUPLICATE_SIMILARITY_THRESHOLD = 0.6

# claim text -> embedding, so re-checking against the same growing list of
# existing claims doesn't re-encode strings we've already embedded once
_claim_embedding_cache: dict[str, np.ndarray] = {}

RESEARCH_NODE_BY_AREA = {
    "overview": "research_overview",
    "industry": "research_industry",
    "financials": "research_financials",
    "news": "research_news",
}


class ClaimCritiquePayload(TypedDict):
    claim: str
    sources: list[dict]


class ResearchState(TypedDict):
    company_name: str
    focus_area: str | None
    # each research node writes only its own category, but on a follow-up
    # run the node returns just the freshly fetched results and the add
    # reducer appends them onto the checkpointed list from the prior run
    overview: Annotated[list[dict], operator.add]
    industry: Annotated[list[dict], operator.add]
    financials: Annotated[list[dict], operator.add]
    news: Annotated[list[dict], operator.add]
    all_findings: list[dict]
    new_findings: list[dict]
    claims: Annotated[list[str], operator.add]
    new_claims: list[str]
    claim_results: Annotated[list[dict], operator.add]
    report: dict


def _get_embedding(text: str) -> np.ndarray:
    if text not in _claim_embedding_cache:
        _claim_embedding_cache[text] = _embedding_model.encode(text)
    return _claim_embedding_cache[text]


def _is_duplicate_claim(claim: str, existing_claims: list[str]) -> bool:
    if not existing_claims:
        return False
    claim_embedding = _get_embedding(claim)
    existing_embeddings = np.stack([_get_embedding(c) for c in existing_claims])
    similarities = np.dot(existing_embeddings, claim_embedding) / (
        np.linalg.norm(existing_embeddings, axis=1) * np.linalg.norm(claim_embedding)
    )
    return bool(np.max(similarities) >= DUPLICATE_SIMILARITY_THRESHOLD)


def route_research(state: ResearchState) -> list[str]:
    focus_area = state.get("focus_area")
    if focus_area is None:
        return list(RESEARCH_NODE_BY_AREA.values())
    return [RESEARCH_NODE_BY_AREA[focus_area]]


def research_overview_node(state: ResearchState) -> dict:
    results = research_company_overview(state["company_name"])
    update = {"overview": results}
    if state.get("focus_area") == "overview":
        update["new_findings"] = results
    return update


def research_industry_node(state: ResearchState) -> dict:
    results = research_industry(state["company_name"])
    update = {"industry": results}
    if state.get("focus_area") == "industry":
        update["new_findings"] = results
    return update


def research_financials_node(state: ResearchState) -> dict:
    results = research_financials(state["company_name"])
    update = {"financials": results}
    if state.get("focus_area") == "financials":
        update["new_findings"] = results
    return update


def research_news_node(state: ResearchState) -> dict:
    results = research_news(state["company_name"])
    update = {"news": results}
    if state.get("focus_area") == "news":
        update["new_findings"] = results
    return update


def collect_node(state: ResearchState) -> dict:
    # .get(..., []) rather than direct indexing: a scoped follow-up run only
    # writes 1 of these 4 keys this invocation. The checkpointer restores the
    # other 3 from the prior run in the normal initial-then-followup flow --
    # but a follow-up called on a thread with no prior initial run would
    # otherwise KeyError here instead of just producing an empty findings set
    all_findings = (
        state.get("overview", [])
        + state.get("industry", [])
        + state.get("financials", [])
        + state.get("news", [])
    )
    return {"all_findings": all_findings}


def decompose_node(state: ResearchState) -> dict:
    focus_area = state.get("focus_area")
    source_findings = state["all_findings"] if focus_area is None else state.get("new_findings", [])

    raw_claims = decompose_into_claims(source_findings)
    existing_claims = state.get("claims", [])
    fresh_claims = [c for c in raw_claims if not _is_duplicate_claim(c, existing_claims)]

    if len(fresh_claims) > MAX_CLAIMS:
        print(f"[decompose] {len(fresh_claims)} new claims found, capping to {MAX_CLAIMS} for this run")
    capped = fresh_claims[:MAX_CLAIMS]

    return {"claims": capped, "new_claims": capped}


def route_to_critique(state: ResearchState):
    # Send() fan-out, not a for-loop invoking one shared compiled subgraph
    # N times inside a single node -- LangChain's own community forum flags
    # that pattern as a checkpointing/namespace pitfall for repeated subgraph
    # invokes, and recommends Send() for exactly this "N independent items"
    # shape instead. Each Send runs critique_one_claim_node as its own
    # parallel branch; claim_results' add reducer collects all of them.
    claims = state["new_claims"]
    if not claims:
        return "report"
    return [Send("critique_one_claim", {"claim": claim, "sources": state["all_findings"]}) for claim in claims]


def critique_one_claim_node(state: ClaimCritiquePayload) -> dict:
    result = _claim_graph.invoke({
        "claim": state["claim"],
        "sources": state["sources"],
        "verdict": "",
        "supporting_count": 0,
        "attempts": 0,
    })
    return {"claim_results": [result]}


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


def build_research_graph(checkpointer=None):
    graph = StateGraph(ResearchState)
    graph.add_node("research_overview", research_overview_node)
    graph.add_node("research_industry", research_industry_node)
    graph.add_node("research_financials", research_financials_node)
    graph.add_node("research_news", research_news_node)
    graph.add_node("collect", collect_node)
    graph.add_node("decompose", decompose_node)
    graph.add_node("critique_one_claim", critique_one_claim_node)
    graph.add_node("report", report_node)

    graph.add_conditional_edges(START, route_research, list(RESEARCH_NODE_BY_AREA.values()))

    graph.add_edge("research_overview", "collect")
    graph.add_edge("research_industry", "collect")
    graph.add_edge("research_financials", "collect")
    graph.add_edge("research_news", "collect")

    graph.add_edge("collect", "decompose")
    graph.add_conditional_edges("decompose", route_to_critique, ["critique_one_claim", "report"])
    graph.add_edge("critique_one_claim", "report")
    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer)
