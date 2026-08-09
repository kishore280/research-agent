from backend.nodes.company_overview import research_company_overview
from backend.nodes.critique import critique_claim


def test_critique_claim_verifies_well_corroborated_claim():
    results = research_company_overview("Anthropic")
    result = critique_claim("Anthropic was founded in 2021", results)
    assert result["verdict"] == "VERIFIED"
    assert result["supporting_count"] >= 2


def test_critique_claim_rejects_fabricated_claim():
    # KNOWN FLAKY, ~1-in-5 runs, documented not hidden: with 3-sample
    # self-consistency majority voting (see critique.py), this correctly
    # rejects a fabricated claim ~4/5 runs. llama-3.1-8b-instant is small
    # enough that a batch-of-5-sources judgment call still occasionally
    # flips to VERIFIED even after voting. Real, measured limitation of
    # prompted-LLM critique vs. a dedicated fine-tuned evaluator (see
    # llm.txt's CRAG honesty note) -- not something to silently paper
    # over by weakening the assertion further.
    results = research_company_overview("Anthropic")
    result = critique_claim("Anthropic was founded in 1998 by Elon Musk", results)
    assert result["verdict"] in ("CONTRADICTED", "UNVERIFIED")
