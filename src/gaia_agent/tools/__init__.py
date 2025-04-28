from typing import List
from langchain_core.tools import BaseTool

from .file_tools import SaveContentTool, DownloadFileTool
from .analysis_tools import AnalyzeCsvTool, AnalyzeExcelTool, ExtractImageTextTool
from .search_tools import TavilySearchTool, DuckDuckGoSearchTool, WikipediaSearchTool
from .misc_tools import PythonReplTool, AnalyzeYoutubeMetadataTool


def get_all_tools(tavily_api_key: str | None = None) -> List[BaseTool]:
    """
    Initializes and returns instances of all available tools.
    Handles potential initialization errors gracefully.
    """
    tools = []
    print("--- Initializing All Tools ---")

    # Search Tools
    try:
        tools.append(TavilySearchTool(api_key=tavily_api_key))
        print("✅ Initialized Tavily Search")
    except Exception as e:
        print(f"⚠️ Tavily Search failed ({e}), falling back to DuckDuckGo.")
        try:
            tools.append(DuckDuckGoSearchTool())
            print("✅ Initialized DuckDuckGo Search")
        except Exception as e_ddg:
            print(f"❌ Failed to initialize DuckDuckGo Search: {e_ddg}")

    try:
        tools.append(WikipediaSearchTool())
        print("✅ Initialized Wikipedia Search")
    except Exception as e:
        print(f"❌ Failed to initialize Wikipedia Search: {e}")

    # File Tools
    try:
        tools.append(SaveContentTool())
        tools.append(DownloadFileTool())
        print("✅ Initialized File Tools (Save, Download)")
    except Exception as e:
        print(f"❌ Failed to initialize File Tools: {e}")

    # Analysis Tools
    try:
        tools.append(AnalyzeCsvTool())
        print("✅ Initialized CSV Analysis Tool")
    except Exception as e:
        print(f"⚠️ CSV Analysis Tool failed: {e}")
    try:
        tools.append(AnalyzeExcelTool())
        print("✅ Initialized Excel Analysis Tool")
    except Exception as e:
        print(f"⚠️ Excel Analysis Tool failed: {e}")
    try:
        tools.append(ExtractImageTextTool())
        print("✅ Initialized Image OCR Tool (requires Tesseract)")
    except Exception as e:
        print(f"⚠️ Image OCR Tool failed: {e}")

    # Misc Tools
    try:
        tools.append(PythonReplTool())
        print("✅ Initialized Python REPL Tool")
    except Exception as e:
        print(f"⚠️ Python REPL Tool failed: {e}")
    try:
        tools.append(AnalyzeYoutubeMetadataTool())
        print("✅ Initialized YouTube Metadata Tool")
    except Exception as e:
        print(f"⚠️ YouTube Metadata Tool failed: {e}")

    print(f"--- Total tools initialized: {len(tools)} ---")
    print(f"Available tool names: {[tool.name for tool in tools]}")
    return tools
