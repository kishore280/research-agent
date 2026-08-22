import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend.graph import RESEARCH_NODE_BY_AREA, build_research_graph
from backend.nodes.classify_focus_area import classify_focus_area

DB_PATH = "research_checkpoints.sqlite"

EMPTY_STATE = {
    "overview": [], "industry": [], "financials": [], "news": [],
    "all_findings": [], "new_findings": [],
    "claims": [], "new_claims": [],
    "claim_results": [], "report": {},
}


class ResearchRequest(BaseModel):
    company_name: str


class FollowupRequest(BaseModel):
    request: str


class ClaimReport(BaseModel):
    claim: str
    verdict: str
    supporting_count: int
    attempts: int


class Report(BaseModel):
    company: str
    claims: list[ClaimReport]


class ResearchResponse(BaseModel):
    thread_id: str
    report: Report


class FollowupResponse(BaseModel):
    focus_area: str
    report: Report


@asynccontextmanager
async def lifespan(app: FastAPI):
    # one long-lived connection/graph for the app's lifetime, not one per
    # request -- SqliteSaver persists state across requests (and process
    # restarts), which is the whole point of Checkpoint 6's follow-up loop.
    # Read directly as app.state.graph in each handler below, not injected
    # via Depends() -- confirmed against FastAPI's own lifespan docs, whose
    # canonical example for this exact case (expensive resource loaded at
    # startup, read in handlers) does the same direct-access thing. Depends()
    # earns its cost mainly for swapping the resource out in tests via
    # app.dependency_overrides, which isn't a goal here (this project tests
    # against real APIs on purpose, not mocks).
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app.state.graph = build_research_graph(checkpointer=checkpointer)
        yield


app = FastAPI(lifespan=lifespan)

# runs on Vite's default port, backend on FastAPI's default port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # a real Groq/Tavily failure (rate limit, timeout, etc.) mid-graph
    # shouldn't leak a raw traceback to the client as a bare 500
    return JSONResponse(status_code=502, content={"detail": f"Upstream research pipeline failed: {exc}"})


@app.post("/research", response_model=ResearchResponse)
async def research(body: ResearchRequest):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"company_name": body.company_name, "focus_area": None, **EMPTY_STATE}

    # graph.invoke() is sync and does real, possibly minutes-long network
    # calls (Tavily + Groq) -- run_in_threadpool so it doesn't block the
    # event loop for every other request while one company's research runs
    result = await run_in_threadpool(app.state.graph.invoke, initial_state, config)
    return ResearchResponse(thread_id=thread_id, report=result["report"])


@app.post("/research/{thread_id}/followup", response_model=FollowupResponse)
async def followup(thread_id: str, body: FollowupRequest):
    try:
        focus_area = await run_in_threadpool(classify_focus_area, body.request)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    assert focus_area in RESEARCH_NODE_BY_AREA  # classify_focus_area already validates this

    config = {"configurable": {"thread_id": thread_id}}
    result = await run_in_threadpool(app.state.graph.invoke, {"focus_area": focus_area}, config)
    return FollowupResponse(focus_area=focus_area, report=result["report"])
