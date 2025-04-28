from typing import Optional, Type
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper


# --- Input Schemas ---
class SearchInput(BaseModel):
    query: str = Field(description="The search query.")


# --- Tool Classes ---
class TavilySearchTool(BaseTool):
    name: str = "tavily_search_results_json"
    description: str = (
        "A search engine optimized for comprehensive, accurate results. "
        "Useful for questions about current events, facts, and general knowledge."
    )
    args_schema: Type[BaseModel] = SearchInput
    api_wrapper: TavilySearchResults

    def __init__(self, api_key: Optional[str] = None, max_results: int = 5):
        super().__init__()  # Initialize BaseTool
        resolved_api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Tavily API Key not found. Set TAVILY_API_KEY env var or pass api_key."
            )
        self.api_wrapper = TavilySearchResults(
            max_results=max_results, api_key=resolved_api_key
        )

    def _run(self, query: str) -> str:
        try:
            return self.api_wrapper.run(query)
        except Exception as e:
            return f"Error during Tavily search: {e}"


class DuckDuckGoSearchTool(BaseTool):
    name: str = "duckduckgo_search"
    description: str = (
        "A fallback search engine. Use this if Tavily search fails or is unavailable. "
        "Useful for questions about current events, facts, and general knowledge."
    )
    args_schema: Type[BaseModel] = SearchInput
    api_wrapper: DuckDuckGoSearchRun

    def __init__(self):
        super().__init__()
        self.api_wrapper = DuckDuckGoSearchRun()

    def _run(self, query: str) -> str:
        try:
            return self.api_wrapper.run(query)
        except Exception as e:
            return f"Error during DuckDuckGo search: {e}"


class WikipediaSearchTool(BaseTool):
    name: str = "wikipedia_search"
    description: str = (
        "Looks up definitions, facts, and information about concepts, people, places, or events on Wikipedia. "
        "Input should be a topic or question."
    )
    args_schema: Type[BaseModel] = SearchInput
    api_wrapper: WikipediaAPIWrapper

    def __init__(self, top_k_results: int = 3, doc_content_chars_max: int = 4000):
        super().__init__()
        try:
            self.api_wrapper = WikipediaAPIWrapper(
                top_k_results=top_k_results, doc_content_chars_max=doc_content_chars_max
            )
        except ImportError:
            raise ImportError(
                "Wikipedia library not found. Run `pip install wikipedia`."
            )

    def _run(self, query: str) -> str:
        try:
            return self.api_wrapper.run(query)
        except Exception as e:
            # Handle cases where a page might not be found gracefully
            if "Page id" in str(e) and "does not match any pages" in str(e):
                return f"Wikipedia page not found for query: '{query}'"
            elif "DisambiguationError" in str(type(e)):
                options = getattr(e, "options", [])
                return f"Ambiguous query: '{query}'. Wikipedia suggests: {', '.join(options[:5])}..."
            return f"Error during Wikipedia search for '{query}': {e}"
