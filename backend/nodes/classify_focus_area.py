import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=5)
model = "openai/gpt-oss-20b"

VALID_AREAS = ["overview", "industry", "financials", "news"]

PROMPT_TEMPLATE = (
    "A user wants to research a company in more depth on one specific area. "
    "Classify their request into EXACTLY ONE of these categories:\n\n"
    "overview = company background, founders, history, mission\n"
    "industry = market, competitors, sector trends\n"
    "financials = funding, revenue, valuation, investors\n"
    "news = recent announcements, events, press coverage\n\n"
    'Respond with a JSON object of the form {{"area": "overview"}} using '
    "exactly one of: overview, industry, financials, news.\n\n"
    "User request: {request}\n\n"
    "JSON:"
)


def classify_focus_area(request: str) -> str:
    prompt = PROMPT_TEMPLATE.format(request=request)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Model returned no content")
    area = json.loads(content)["area"]
    if area not in VALID_AREAS:
        raise ValueError(f"Model returned an invalid area: {area!r}")
    return area
