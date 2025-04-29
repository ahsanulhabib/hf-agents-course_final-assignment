import os
from typing import Optional, Type
from pydantic import BaseModel, Field

from base_tool import BaseTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv


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

    def __init__(self, **kwargs):
        # Create the api_wrapper instance first
        api_wrapper_instance = DuckDuckGoSearchRun()
        # Pass the created api_wrapper instance to the parent constructor
        # Pydantic's __init__ will set this field correctly.
        super().__init__(api_wrapper=api_wrapper_instance, **kwargs)

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
    api_wrapper: WikipediaAPIWrapper  # Pydantic expects this field to be set

    # Modified __init__ to correctly initialize and pass api_wrapper to the parent
    def __init__(
        self, top_k_results: int = 3, doc_content_chars_max: int = 4000, **kwargs
    ):
        # Create the instance of the api_wrapper *before* calling super()
        try:
            api_wrapper_instance = WikipediaAPIWrapper(
                top_k_results=top_k_results, doc_content_chars_max=doc_content_chars_max
            )
        except ImportError:
            raise ImportError(
                "Wikipedia library not found. Run `pip install wikipedia`."
            )

        # Call the parent constructor and pass the api_wrapper instance along with other kwargs
        # This ensures Pydantic validation in the parent class succeeds because api_wrapper is provided
        super().__init__(api_wrapper=api_wrapper_instance, **kwargs)

        # The api_wrapper is now set via the super().__init__ call, no need to set it again directly

    def _run(self, query: str) -> str:
        try:
            # Use the api_wrapper instance that was set during initialization via super()
            return self.api_wrapper.run(query)
        except Exception as e:
            if "Page id" in str(e) and "does not match any pages" in str(e):
                return f"Wikipedia page not found for query: '{query}'"
            elif "DisambiguationError" in str(type(e)):
                options = getattr(e, "options", [])
                return f"Ambiguous query: '{query}'. Wikipedia suggests: {', '.join(options[:5])}..."
            return f"Error during Wikipedia search for '{query}': {e}"


if __name__ == "__main__":
    # Example usage

    load_dotenv()  # Load environment variables from .env file

    tavily_api_key = os.getenv("TAVILY_API_TOKEN") or os.getenv("TAVILY_API_KEY")

    if tavily_api_key:
        web_search_tool = TavilySearchTool(api_key=tavily_api_key)
    else:
        print(
            "TAVILY_API_TOKEN not found in .env or environment, using DuckDuckGo search tool."
        )
        web_search_tool = DuckDuckGoSearchTool()

    query = "What is the capital of Bangladesh?"

    web_search_result = web_search_tool._run(query)

    print("===" * 50)
    print(web_search_result)
    print("===" * 50)

    wiki_tool = WikipediaSearchTool()

    print("===" * 50)
    print(wiki_tool._run("What is the capital of Bangladesh?"))
    print("===" * 50)
