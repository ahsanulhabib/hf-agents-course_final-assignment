import os
from typing import List, Optional
from langchain_core.tools import BaseTool

# Import tool classes from sibling modules
from gaia_agent.tools.file_tools import SaveContentTool, DownloadFileTool
from gaia_agent.tools.analysis_tools import (
    AnalyzeTextTool,
    AnalyzeCsvTool,
    AnalyzeExcelTool,
    AnalyzeImageContentTool,
    AnalyzeMP3Tool,
    ExtractImageTextTool,
)
from gaia_agent.tools.search_tools import (
    TavilySearchTool,
    DuckDuckGoSearchTool,
    WikipediaSearchTool,
    ArXivSearchTool,
    ArxivDocumentSearchTool,
)
from gaia_agent.tools.misc_tools import (
    PythonReplTool,
    AnalyzeYoutubeMetadataTool,
    AddTool,
    SubtractTool,
    MultiplyTool,
    DivideTool,
    ModulusTool,
)


def get_all_tools(tavily_api_key: Optional[str] = None) -> List[BaseTool]:
    """
    Initializes and returns instances of all available tools.
    Handles potential initialization errors gracefully.
    """
    tools = []
    print("--- Initializing All Tools ---")

    # Search Tools
    tavily_key = (
        tavily_api_key or os.getenv("TAVILY_API_KEY") or os.getenv("TAVILY_API_TOKEN")
    )
    if tavily_key:
        try:
            tools.append(TavilySearchTool(api_key=tavily_key))
            print("✅ Initialized Tavily Search")
        except Exception as e:
            print(f"⚠️ Tavily Search failed ({e}), falling back to DuckDuckGo.")
            try:
                tools.append(DuckDuckGoSearchTool())
                print("✅ Initialized DuckDuckGo Search (Fallback)")
            except Exception as e_ddg:
                print(f"❌ Failed to initialize DuckDuckGo Search: {e_ddg}")
    else:
        try:
            tools.append(DuckDuckGoSearchTool())
            print("✅ Initialized DuckDuckGo Search (Primary)")
        except Exception as e_ddg:
            print(f"❌ Failed to initialize DuckDuckGo Search: {e_ddg}")

    try:
        tools.append(WikipediaSearchTool())
        print("✅ Initialized Wikipedia Search")
    except Exception as e:
        print(f"❌ Failed to initialize Wikipedia Search: {e}")

    try:
        tools.append(ArXivSearchTool())
        print("✅ Initialized ArXiv Search")
    except Exception as e:
        print(f"❌ Failed to initialize ArXiv Search: {e}")

    try:
        tools.append(ArxivDocumentSearchTool())
        print("✅ Initialized ArXiv Document Search")
    except Exception as e:
        print(f"❌ Failed to initialize ArXiv Document Search: {e}")

    # File Tools
    try:
        tools.append(SaveContentTool())
        tools.append(DownloadFileTool())
        print("✅ Initialized File Tools (Save, Download)")
    except Exception as e:
        print(f"❌ Failed to initialize File Tools: {e}")

    # Analysis Tools
    try:
        tools.append(AnalyzeTextTool())
        print("✅ Initialized Text Analysis Tool")
    except Exception as e:
        print(f"❌ Failed to initialize Analysis Tool: {e}")
    try:
        tools.append(AnalyzeCsvTool())
        print("✅ Initialized CSV Analysis Tool")
    except Exception as e:
        print(f"❌ Failed to initialize Analysis Tool: {e}")
    try:
        tools.append(AnalyzeExcelTool())
        print("✅ Initialized Excel Analysis Tool")
    except Exception as e:
        print(f"❌ Failed to initialize Analysis Tool: {e}")
    try:
        tools.append(AnalyzeImageContentTool())
        print("✅ Initialized Image Analysis Tool")
    except Exception as e:
        print(f"❌ Failed to initialize Analysis Tool: {e}")
    try:
        tools.append(AnalyzeMP3Tool())
        print("✅ Initialized MP3 Analysis Tool")
    except Exception as e:
        print(f"❌ Failed to initialize Analysis Tool: {e}")
    try:
        tools.append(ExtractImageTextTool())
        print("✅ Initialized Image OCR Tool")
    except Exception as e:
        print(f"❌ Failed to initialize Analysis Tool: {e}")

    # Misc Tools
    try:
        tools.append(PythonReplTool())
        print("✅ Initialized Python REPL Tool")
    except Exception as e:
        print(f"❌ Failed to initialize Python REPL Tool: {e}")
    try:
        tools.append(AnalyzeYoutubeMetadataTool())
        print("✅ Initialized YouTube Metadata Tool")
    except Exception as e:
        print(f"❌ Failed to initialize Analysis Tool: {e}")

    try:
        tools.append(AddTool())
        tools.append(SubtractTool())
        tools.append(MultiplyTool())
        tools.append(DivideTool())
        tools.append(ModulusTool())
        print("✅ Initialized Math Tools (Add, Subtract, Multiply, Divide, Modulus)")
    except Exception as e:
        print(f"❌ Failed to initialize Math Tools: {e}")

    print(f"--- Total tools initialized: {len(tools)} ---")
    print(f"Available tool names: {[tool.name for tool in tools]}")
    return tools


if __file__ == "__main__":
    # Example usage of the get_all_tools function
    all_tools = get_all_tools()
    print(f"Initialized {len(all_tools)} tools.")
    for tool in all_tools:
        print(f"Tool Name: {tool.name}, Description: {tool.description}")
    print("All tools initialized successfully.")
