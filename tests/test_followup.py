import os

from langgraph.checkpoint.sqlite import SqliteSaver

from backend.graph import build_research_graph

# real end-to-end test against live Tavily + Groq APIs -- expensive
# (full initial run + a scoped follow-up run), only run manually when
# Groq's daily token quota has headroom, not part of the default suite
INITIAL_STATE = {
    "company_name": "Anthropic",
    "focus_area": None,
    "overview": [], "industry": [], "financials": [], "news": [],
    "all_findings": [], "new_findings": [],
    "claims": [], "new_claims": [],
    "claim_results": [], "report": {},
}

# SqliteSaver over MemorySaver: state survives a process restart between
# the initial run and a later follow-up request, which is the realistic
# shape once this is behind a FastAPI service (Checkpoint 7) -- MemorySaver
# would lose everything the moment the process handling the initial
# request exits
DB_PATH = "test_followup_checkpoints.sqlite"


def test_followup_scoped_to_financials_appends_without_reverifying():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    config = {"configurable": {"thread_id": "anthropic-followup-test"}}

    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = build_research_graph(checkpointer=checkpointer)
        result1 = graph.invoke(INITIAL_STATE, config)

    initial_claim_count = len(result1["claims"])
    assert initial_claim_count > 0

    # re-open the same db in a fresh SqliteSaver instance -- proves the
    # follow-up doesn't depend on reusing the same in-memory Python object
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = build_research_graph(checkpointer=checkpointer)
        result2 = graph.invoke({"focus_area": "financials"}, config)

    # old claims untouched, new ones appended -- not re-verified, not dropped
    assert result2["claims"][:initial_claim_count] == result1["claims"]
    assert len(result2["claims"]) > initial_claim_count
    assert len(result2["claim_results"]) == len(result2["claims"])

    # categories outside the focus area weren't re-fetched
    assert result2["overview"] == result1["overview"]
    assert result2["industry"] == result1["industry"]
    assert result2["news"] == result1["news"]
    assert len(result2["financials"]) > len(result1["financials"])

    os.remove(DB_PATH)
