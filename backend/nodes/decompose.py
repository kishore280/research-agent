import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=5)
model = "openai/gpt-oss-20b"

PROMPT_TEMPLATE = (
    "Given the following research findings about a company, extract a list "
    "of individual, checkable factual claims (e.g. founding year, founders, "
    "headquarters, funding amount, employee count, mission/focus area). "
    "Each claim should be a short, standalone statement of fact.\n\n"
    'Respond with a JSON object of the form {{"claims": ["claim one", '
    '"claim two", ...]}}.\n\n'
    "Findings:\n{findings}\n\n"
    "JSON:"
)

# ilana error adiku
MAX_CONTENT_CHARS_PER_FINDING = 500


def decompose_into_claims(findings: list[dict]) -> list[str]:
    findings_text = " ".join(f["content"][:MAX_CONTENT_CHARS_PER_FINDING] for f in findings)
    prompt = PROMPT_TEMPLATE.format(findings=findings_text)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Model returned no content")
    claims = json.loads(content)["claims"]
    return claims
