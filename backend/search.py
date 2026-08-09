import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

MAX_RESULTS = 5
SEARCH_ERROR = "No search results found"


def search(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    # call client.search(query=query, max_results=max_results)
    response = client.search(query=query, max_results=max_results)
    if not response["results"]:
        return [{"error": SEARCH_ERROR}]
    return response["results"]


