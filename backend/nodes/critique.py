import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
model = "llama-3.1-8b-instant"

PROMPT_TEMPLATE = (
    "Claim to check: {claim}\n\n"
    "Below are numbered sources. For EACH source, decide if it SUPPORTS "
    "the claim, CONTRADICTS the claim, or is IRRELEVANT to the claim.\n\n"
    "{sources}\n\n"
    'Respond with a JSON object: {{"verdicts": ["SUPPORTS", "IRRELEVANT", '
    '...]}} -- one verdict per source, in the same order as listed above.'
)


def critique_claim(claim: str, sources: list[dict]) -> dict:
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
    supporting_count = sum(1 for v in verdicts if v == "SUPPORTS")
    verdict = "VERIFIED" if supporting_count >= 2 else "UNVERIFIED"
    if "CONTRADICTS" in verdicts and verdict != "VERIFIED":
        verdict = "CONTRADICTED"
    return {
        "claim": claim,
        "verdict": verdict,
        "supporting_count": supporting_count,
        "source_verdicts": list(zip(sources, verdicts)),
    }
