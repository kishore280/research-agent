import json
import os
from collections import Counter

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# need more headroom than the default 2 retries for our call volume.
client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=5)
model = "openai/gpt-oss-20b"

PROMPT_TEMPLATE = (
    "Claim to check: {claim}\n\n"
    "Below are numbered sources. For EACH source, decide if it SUPPORTS "
    "the claim, CONTRADICTS the claim, or is IRRELEVANT to the claim.\n\n"
    "{sources}\n\n"
    'Respond with a JSON object: {{"verdicts": ["SUPPORTS", "IRRELEVANT", '
    '...]}} -- one verdict per source, in the same order as listed above.'
)

SELF_CONSISTENCY_SAMPLES = 3


def _sample_source_verdicts(claim: str, sources: list[dict]) -> list[str] | None:
    sources_text = "\n\n".join(f"{i+1}. {s['url']}\n{s['content']}" for i, s in enumerate(sources))
    prompt = PROMPT_TEMPLATE.format(claim=claim, sources=sources_text)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Model returned no content")
    verdicts = json.loads(content)["verdicts"]
    # a bad sample (wrong length) is worse than no sample -- drop it
    # rather than let it corrupt the majority vote with misaligned indices
    if len(verdicts) != len(sources):
        return None
    return verdicts


def critique_claim(claim: str, sources: list[dict]) -> dict:
    if not sources:
        return {"claim": claim, "verdict": "UNVERIFIED", "supporting_count": 0, "source_verdicts": []}

    # self-consistency: sample the same judgment multiple independent
    # times and take a majority vote per source, instead of trusting one
    # noisy call -- a single small-model sample flipped a fabricated claim
    # ("Anthropic founded in 1998 by Elon Musk") to VERIFIED in testing
    samples = [_sample_source_verdicts(claim, sources) for _ in range(SELF_CONSISTENCY_SAMPLES)]
    valid_samples = [s for s in samples if s is not None]
    if not valid_samples:
        raise RuntimeError("All self-consistency samples were malformed")

    majority_verdicts = []
    for i in range(len(sources)):
        votes = [sample[i] for sample in valid_samples]
        majority_verdicts.append(Counter(votes).most_common(1)[0][0])

    supporting_count = sum(1 for v in majority_verdicts if v == "SUPPORTS")
    verdict = "VERIFIED" if supporting_count >= 2 else "UNVERIFIED"
    if "CONTRADICTS" in majority_verdicts and verdict != "VERIFIED":
        verdict = "CONTRADICTED"

    return {
        "claim": claim,
        "verdict": verdict,
        "supporting_count": supporting_count,
        "source_verdicts": list(zip(sources, majority_verdicts)),
    }
