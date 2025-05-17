import os
import json
from typing import Type, Optional, Any
from pydantic import BaseModel, Field

from langchain_community.tools.tavily_search import (
    TavilySearchResults as LC_TavilySearchTool,
)
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.arxiv import ArxivAPIWrapper

# Import Arxiv Loader
try:
    from langchain_community.document_loaders import ArxivLoader
except ImportError:
    ArxivLoader = None

from dotenv import load_dotenv

from gaia_agent.tools.base_tool import BaseTool
from gaia_agent.logger_config import logger


# --- Input Schemas ---
class SearchInput(BaseModel):
    query: str = Field(description="The search query.")


# --- Tool Classes ---
class TavilySearchTool(BaseTool):
    name: str = "tavily_search"  # Name of the tool
    description: str = (  # Description explaining the tool's purpose and input
        "A search engine optimized for comprehensive, accurate results. "
        "Useful for questions about current events, facts, and general knowledge. "
        "Input is a search query."
    )
    args_schema: Type[BaseModel] = SearchInput  # Schema for the tool's input
    api_wrapper: TavilySearchAPIWrapper  # Field to hold the API wrapper instance
    max_results: int = 10  # Field to hold the maximum number of search results

    def __init__(
        self, api_key: Optional[str] = None, max_results: int = 10, **kwargs: Any
    ):
        # Resolve the Tavily API key from argument or environment variables
        resolved_api_key = (
            api_key or os.getenv("TAVILY_API_TOKEN") or os.getenv("TAVILY_API_KEY")
        )
        if not resolved_api_key:
            # Raise error if API key is not found
            raise ValueError("Tavily API Key not found. Set your Tavily API key.")

        # Create the TavilySearchAPIWrapper instance before initializing the base class
        try:
            tavily_wrapper = TavilySearchAPIWrapper(tavily_api_key=resolved_api_key)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize TavilySearchAPIWrapper: {e}"
            ) from e

        # Initialize the BaseTool class, passing all required fields
        super().__init__(
            name="tavily_search",
            # description=self.description,
            # args_schema=self.args_schema,
            api_wrapper=tavily_wrapper,
            max_results=max_results,
            **kwargs,
        )

    def _run(self, query: str, run_manager: Optional[Any] = None) -> str:
        """Executes the Tavily search using the API wrapper."""
        try:
            # Call the results method on the stored api_wrapper instance
            results = self.api_wrapper.results(query, max_results=self.max_results)
            # Convert the results to a string (e.g., JSON) for the tool's output
            return json.dumps(results, indent=2)
        except Exception as e:
            # Return an error message if the search fails
            return f"Error performing Tavily search for '{query}': {e}"


class DuckDuckGoSearchTool(BaseTool):
    name: str = "duckduckgo_search"
    description: str = (
        "Use this for ALL web searches, including music, lyrics, and general facts. "
        "Preferred over Wikipedia for up-to-date or broad information."
    )
    args_schema: Type[BaseModel] = SearchInput
    api_wrapper: DuckDuckGoSearchRun

    def __init__(self, **kwargs):
        # Create the api_wrapper instance first
        api_wrapper_instance = DuckDuckGoSearchRun()
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
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max,
                wiki_client=None,  # Set wiki_client to None or your desired client instance
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


class ArXivSearchTool(BaseTool):
    name: str = "arxiv_search"
    description: str = (
        "Searches scientific papers on arXiv. "
        "Useful for finding research papers, abstracts, and scientific information."
    )
    args_schema: Type[BaseModel] = SearchInput
    api_wrapper: ArxivAPIWrapper

    def __init__(self, top_k_results: int = 10, **kwargs):
        api_wrapper_instance = ArxivAPIWrapper(
            top_k_results=top_k_results,
            # arxiv_search=arxiv_search,
            # arxiv_exceptions=arxiv_exceptions,
        )
        super().__init__(api_wrapper=api_wrapper_instance, **kwargs)

    def _run(self, query: str) -> str:
        try:
            return self.api_wrapper.run(query)
        except Exception as e:
            return f"Error during arXiv search: {e}"


class ArxivDocumentSearchTool(BaseTool):
    name: str = "arxiv_doc_search"
    description: str = (
        "Searches ArXiv for scientific papers. Useful for finding research papers, abstracts, and authors. "
        "Input should be a search query (e.g., title keywords, author name)."
    )
    args_schema: Type[BaseModel] = SearchInput

    def __init__(self, load_max_docs: int = 3, doc_content_chars_max: int = 4000):
        super().__init__()
        if ArxivLoader is None:
            raise ImportError(
                "Arxiv tool requires `arxiv` library. Run `pip install arxiv`."
            )
        self.load_max_docs = load_max_docs
        self.doc_content_chars_max = doc_content_chars_max

    def _run(self, query: str) -> str:
        logger.debug(f"Running Arxiv Search for: {query}")
        try:
            if ArxivLoader is None:
                raise ImportError(
                    "Arxiv tool requires `arxiv` library. Run `pip install arxiv`."
                )
            loader = ArxivLoader(
                query=query,
                load_max_docs=self.load_max_docs,
                # load_all_available_meta=True # Optionally load more metadata
            )
            search_docs = loader.load()

            if not search_docs:
                return f"No documents found on ArXiv for query: '{query}'"

            # Format results similar to other search tools
            formatted_results = []
            for doc in search_docs:
                # Truncate content
                content = doc.page_content[: self.doc_content_chars_max]
                if len(doc.page_content) > self.doc_content_chars_max:
                    content += "..."
                # Safely access metadata
                source = doc.metadata.get("entry_id", "N/A")  # Arxiv uses entry_id
                title = doc.metadata.get("Title", "N/A")
                published = doc.metadata.get("Published", "N/A")  # Publication date

                formatted_results.append(
                    f'<Document source="{source}" title="{title}" published="{published}">\n{content}\n</Document>'
                )

            return "\n\n---\n\n".join(formatted_results)

        except Exception as e:
            logger.exception(f"Error during Arxiv search for '{query}'")
            return f"Error during Arxiv search: {e}"


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

    query = "Tracklist for 'Hybrid Theory' by Linkin Park"

    print("===" * 50)
    print(web_search_tool._run(query))
    print("===" * 50)

    wiki_tool = WikipediaSearchTool()

    print("===" * 50)
    print(wiki_tool._run(query))
    print("===" * 50)

    arxiv_tool = ArXivSearchTool()
    print("===" * 50)
    print(arxiv_tool._run("quantum computing"))
    print("===" * 50)
