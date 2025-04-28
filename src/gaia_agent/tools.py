import os
import tempfile
import re
import json
import requests
import io
import time
import contextlib
from dotenv import load_dotenv
import yt_dlp
from urllib.parse import urlparse
from typing import List, Optional, Any, Union

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import BaseTool, Tool, tool

# Load environment variables from .env file
load_dotenv()

# --- GAIA-specific Tools ---


@tool
def save_content_to_file(content: str, filename: Optional[str] = None) -> str:
    """
    Saves the given text content to a temporary file and returns the file path.
    Use this when you need to make text content available as a file for other tools (like analysis tools).

    Args:
        content: The text content to save.
        filename: Optional desired filename. A random name will be generated if not provided.

    Returns:
        A message indicating the path where the file was saved, e.g., "File saved to /tmp/xyz.txt".
    """
    try:
        temp_dir = tempfile.gettempdir()
        if filename:
            filename = os.path.basename(filename)  # Basic sanitization
            filepath = os.path.join(temp_dir, filename)
        else:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, mode="w", suffix=".txt", encoding="utf-8"
            )
            filepath = temp_file.name
            temp_file.close()

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Content successfully saved to temporary file: {filepath}"
    except Exception as e:
        return f"Error saving content to file: {str(e)}"


@tool
def download_file_from_url(url: str, filename: Optional[str] = None) -> str:
    """
    Downloads a file from a given URL and saves it to a temporary location.
    Handles common file types based on URL or content.

    Args:
        url: The URL of the file to download.
        filename: Optional desired filename for the saved file. If None, it tries to infer from the URL.

    Returns:
        A message indicating the path to the downloaded file or an error message.
    """
    try:
        temp_dir = tempfile.gettempdir()
        if not filename:
            try:
                parsed_path = urlparse(url).path
                filename = os.path.basename(parsed_path) if parsed_path else None
                if not filename:
                    with requests.get(url, stream=True, timeout=30) as r_head:
                        r_head.raise_for_status()
                        cd = r_head.headers.get("content-disposition")
                        if cd:
                            fname = re.findall('filename="?(.+)"?', cd)
                            if fname:
                                filename = fname[0]
                if not filename:
                    import uuid

                    content_type = (
                        requests.head(url, timeout=10)
                        .headers.get("content-type", "")
                        .split(";")[0]
                    )
                    extension_map = {
                        "application/pdf": ".pdf",
                        "image/jpeg": ".jpg",
                        "image/png": ".png",
                        "text/csv": ".csv",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                    }
                    ext = extension_map.get(content_type, ".download")
                    filename = f"downloaded_{uuid.uuid4().hex[:8]}{ext}"
            except Exception as e:
                print(
                    f"Warning: Could not reliably determine filename from URL {url}: {e}. Generating random name."
                )
                import uuid

                filename = f"downloaded_{uuid.uuid4().hex[:8]}.download"

        filename = os.path.basename(filename)  # Sanitize again
        filepath = os.path.join(temp_dir, filename)

        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return f"File successfully downloaded from {url} and saved to temporary path: {filepath}"
    except requests.exceptions.RequestException as e:
        return f"Error downloading file from URL {url}: Network error - {str(e)}"
    except Exception as e:
        return f"Error downloading file from URL {url}: {str(e)}"


@tool
def extract_text_from_image(image_path: str) -> str:
    """
    Extracts text content from an image file using Optical Character Recognition (OCR).
    Requires the image file to exist locally (e.g., downloaded previously).

    Args:
        image_path: The local file path to the image file (e.g., /tmp/my_image.png).

    Returns:
        The extracted text as a string, or an error message if OCR fails or is unavailable.
    """
    try:
        from PIL import Image
        import pytesseract

        if not os.path.exists(image_path):
            return f"Error: Image file not found at path: {image_path}"
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            return "Error: Tesseract OCR engine not found or not configured correctly. Please ensure it's installed and in your system's PATH."

        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)

        if not text.strip():
            return f"Successfully processed image '{os.path.basename(image_path)}', but no text was detected."
        else:
            return f"Extracted text from image '{os.path.basename(image_path)}':\n---\n{text}\n---"
    except ImportError:
        return "Error: Required libraries for image OCR (Pillow, pytesseract) are not installed. Run `pip install Pillow pytesseract`."
    except FileNotFoundError:
        return "Error: Tesseract OCR engine not found or not configured correctly. Please ensure it's installed and in your system's PATH."
    except Exception as e:
        if "TesseractNotFoundError" in str(type(e)):
            return "Error: Tesseract OCR engine not found or not configured correctly. Please ensure it's installed and in your system's PATH."
        return f"Error extracting text from image '{os.path.basename(image_path)}': {str(e)}"


@tool
def analyze_csv_file(file_path: str, query: str) -> str:
    """
    Analyzes a CSV file using pandas to answer a specific query about its content.
    Provides basic info (rows, columns, headers) and attempts to answer the query based on the data.

    Args:
        file_path: The local path to the CSV file (e.g., /tmp/data.csv).
        query: The specific question to answer about the CSV data (e.g., "What is the average value in the 'Sales' column?").

    Returns:
        A summary of the CSV and the answer to the query, or an error message.
    """
    try:
        import pandas as pd

        if not os.path.exists(file_path):
            return f"Error: CSV file not found at path: {file_path}"

        df = pd.read_csv(file_path)
        num_rows, num_cols = df.shape
        columns = ", ".join(df.columns)
        result = f"CSV file '{os.path.basename(file_path)}' loaded: {num_rows} rows, {num_cols} columns.\nColumns: {columns}\n\n"
        result += f"Query: '{query}'\n\n"
        result += "First 5 rows:\n" + df.head().to_string() + "\n\n"
        result += (
            "Basic Statistics (describe):\n"
            + df.describe(include="all").to_string()
            + "\n\n"
        )

        # Simple heuristic based query answering (can be expanded)
        if "average" in query.lower() or "mean" in query.lower():
            col_match = re.search(r"['\"](.*?)['\"]", query)
            if col_match:
                col_name = col_match.group(1)
                if col_name in df.columns and pd.api.types.is_numeric_dtype(
                    df[col_name]
                ):
                    mean_val = df[col_name].mean()
                    result += f"Answer based on query heuristic: The mean of column '{col_name}' is {mean_val:.2f}\n"
                else:
                    result += f"Could not calculate mean for column '{col_name}' (not found or not numeric).\n"

        result += (
            "\nUse the Python REPL tool for more complex analysis if needed, using the file path: "
            + file_path
        )
        return result
    except ImportError:
        return "Error: pandas library is not installed. Run `pip install pandas`."
    except Exception as e:
        return f"Error analyzing CSV file '{os.path.basename(file_path)}': {str(e)}"


@tool
def analyze_excel_file(
    file_path: str, query: str, sheet_name: Optional[Union[str, int]] = 0
) -> str:
    """
    Analyzes an Excel file (.xls, .xlsx) using pandas to answer a specific query about its content.
    Provides basic info (rows, columns, headers) for a specified sheet and attempts to answer the query.

    Args:
        file_path: The local path to the Excel file (e.g., /tmp/data.xlsx).
        query: The specific question to answer about the data (e.g., "How many entries are in the 'Status' column?").
        sheet_name: The name or index of the sheet to analyze (default is the first sheet, index 0).

    Returns:
        A summary of the specified Excel sheet and the answer to the query, or an error message.
    """
    try:
        import pandas as pd

        if not os.path.exists(file_path):
            return f"Error: Excel file not found at path: {file_path}"
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as read_e:
            try:
                xls = pd.ExcelFile(file_path)
                available_sheets = xls.sheet_names
                return f"Error reading sheet '{sheet_name}' from Excel file '{os.path.basename(file_path)}': {read_e}. Available sheets: {available_sheets}"
            except Exception as inner_e:
                return f"Error reading Excel file '{os.path.basename(file_path)}': {read_e} (also failed to list sheets: {inner_e})"

        num_rows, num_cols = df.shape
        columns = ", ".join(df.columns)
        result = f"Excel file '{os.path.basename(file_path)}', Sheet '{sheet_name}' loaded: {num_rows} rows, {num_cols} columns.\nColumns: {columns}\n\n"
        result += f"Query: '{query}'\n\n"
        result += "First 5 rows:\n" + df.head().to_string() + "\n\n"
        result += (
            "Basic Statistics (describe):\n"
            + df.describe(include="all").to_string()
            + "\n\n"
        )

        if "count" in query.lower() or "how many" in query.lower():
            col_match = re.search(r"['\"](.*?)['\"]", query)
            if col_match:
                col_name = col_match.group(1)
                if col_name in df.columns:
                    count_val = df[col_name].count()
                    result += f"Answer based on query heuristic: The count of non-missing entries in column '{col_name}' is {count_val}\n"
                else:
                    result += f"Could not count entries for column '{col_name}' (not found).\n"

        result += (
            "\nUse the Python REPL tool for more complex analysis if needed, using the file path: "
            + file_path
            + f" and sheet_name='{sheet_name}'"
        )
        return result
    except ImportError:
        return "Error: pandas and openpyxl libraries are not installed. Run `pip install pandas openpyxl`."
    except Exception as e:
        return f"Error analyzing Excel file '{os.path.basename(file_path)}': {str(e)}"


@tool
def analyze_youtube_metadata(youtube_url: str) -> str:
    """
    Extracts metadata (title, description) from a YouTube video URL using yt-dlp
    and provides a summary based on that metadata. This tool does NOT watch the video.

    Args:
        youtube_url: The full URL of the YouTube video (e.g., https://www.youtube.com/watch?v=...).

    Returns:
        A summary based on the video's metadata or an error message.
    """
    print(f"--- Tool: analyze_youtube_metadata ({youtube_url}) ---")
    try:
        parsed_url = urlparse(youtube_url)
        if not all([parsed_url.scheme, parsed_url.netloc]) or (
            "youtube.com" not in parsed_url.netloc
            and "youtu.be" not in parsed_url.netloc
        ):
            return "Error: Please provide a valid YouTube video URL (e.g., https://www.youtube.com/watch?v=...)."

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": "in_playlist",
            "forcejson": True,
            "skip_download": True,
            "youtube_include_dash_manifest": False,
        }
        extracted_info_json = None
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    ydl.extract_info(youtube_url, download=False)
                extracted_info_json = buffer.getvalue()
                if not extracted_info_json:
                    return f"Error: yt-dlp did not return metadata for {youtube_url}."

                first_video_info = None
                for line in extracted_info_json.strip().split("\n"):
                    try:
                        first_video_info = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
                if not first_video_info:
                    return f"Error: Could not parse metadata JSON from yt-dlp for {youtube_url}."

                title = first_video_info.get("title", "N/A")
                description = first_video_info.get("description", "N/A")
                uploader = first_video_info.get("uploader", "N/A")
                duration_seconds = first_video_info.get("duration")
                duration_str = (
                    time.strftime("%H:%M:%S", time.gmtime(duration_seconds))
                    if duration_seconds
                    else "N/A"
                )
                view_count = first_video_info.get("view_count", "N/A")
                upload_date = first_video_info.get("upload_date", "N/A")
                if upload_date and len(upload_date) == 8:
                    upload_date = (
                        f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
                    )

                summary = (
                    f"Metadata analysis for YouTube video: {youtube_url}\n"
                    f"Title: {title}\nUploader: {uploader}\nUpload Date: {upload_date}\n"
                    f"Duration: {duration_str}\nView Count: {view_count}\n"
                    f"Description (truncated):\n---\n{description[:1000]}{'...' if len(description) > 1000 else ''}\n---\n"
                    "Note: This analysis is based *only* on the video's metadata."
                )
                return summary
            except yt_dlp.utils.DownloadError as e:
                if "age restricted" in str(e).lower() or "sign in" in str(e).lower():
                    return f"Error: Cannot access metadata for {youtube_url}. Video likely age-restricted or requires login."
                elif "video unavailable" in str(e).lower():
                    return f"Error: Video at {youtube_url} is unavailable."
                else:
                    return f"Error: yt-dlp failed to extract metadata for {youtube_url}. Details: {str(e)}"
            except Exception as e:
                return f"Error: Unexpected error during metadata extraction for {youtube_url}: {str(e)}"
    except Exception as e:
        return f"Error setting up YouTube metadata analysis for {youtube_url}: {str(e)}"


# --- Tool Getter Function ---


def get_tools(tavily_api_key: Optional[str] = None) -> List[BaseTool]:
    """
    Initializes and returns a list of tools for the agent.
    """
    tools = []
    print("--- Initializing Tools ---")

    # 1. Search Tool (Tavily preferred, DDG fallback)
    tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
    if tavily_key:
        print("Initializing Tavily Search Tool...")
        try:
            tools.append(
                TavilySearchResults(
                    max_results=5,
                    api_key=tavily_key,
                    description="Search engine for current events, facts, general knowledge. Input: search query.",
                )
            )
        except Exception as e:
            print(
                f"Warning: Failed to initialize Tavily: {e}. Falling back to DuckDuckGo."
            )
            tools.append(
                DuckDuckGoSearchRun(
                    description="Search engine for current events, facts, general knowledge. Input: search query."
                )
            )
    else:
        print("Tavily API Key not found. Initializing DuckDuckGo Search Tool...")
        tools.append(
            DuckDuckGoSearchRun(
                description="Search engine for current events, facts, general knowledge. Input: search query."
            )
        )

    # 2. Python REPL Tool
    print("Initializing Python REPL Tool...")
    try:
        tools.append(
            PythonREPLTool(
                description="Python shell for calculations, data manipulation, code execution, file checks (os.path.exists). ALWAYS print() results. Input: valid Python code."
            )
        )
    except ImportError:
        print(
            "Warning: Python REPL Tool failed. `lark-parser` missing? (`pip install lark-parser`)."
        )
    except Exception as e:
        print(f"Warning: Failed to initialize Python REPL Tool: {e}")

    # 3. Wikipedia Tool
    print("Initializing Wikipedia Search Tool...")
    try:
        wikipedia_api = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=4000)
        tools.append(
            Tool(
                name="wikipedia_search",
                func=wikipedia_api.run,
                description="Looks up definitions, facts on Wikipedia. Input: topic or question.",
            )
        )
    except ImportError:
        print(
            "Warning: `wikipedia` library not installed? (`pip install wikipedia`). Wikipedia tool unavailable."
        )
    except Exception as e:
        print(f"Warning: Failed to initialize Wikipedia Tool: {e}")

    # 4. GAIA-specific tools (File Handling, Analysis, OCR)
    print("Initializing GAIA-specific tools (File, CSV, Excel, Image)...")
    tools.append(save_content_to_file)
    tools.append(download_file_from_url)
    try:
        import pandas

        tools.append(analyze_csv_file)
        try:
            import openpyxl

            tools.append(analyze_excel_file)
        except ImportError:
            print("Warning: `openpyxl` not installed? Excel analysis tool unavailable.")
    except ImportError:
        print("Warning: `pandas` not installed? CSV/Excel analysis tools unavailable.")
    try:
        import PIL
        import pytesseract

        try:
            pytesseract.get_tesseract_version()
            tools.append(extract_text_from_image)
        except Exception:
            print(
                "Warning: Tesseract OCR engine not found/configured? Image text extraction may fail."
            )
            tools.append(extract_text_from_image)
    except ImportError:
        print(
            "Warning: `Pillow` or `pytesseract` not installed? Image text extraction unavailable."
        )

    # 5. YouTube Metadata Tool
    print("Initializing YouTube Metadata Analysis Tool...")
    try:
        import yt_dlp

        tools.append(analyze_youtube_metadata)
    except ImportError:
        print("Warning: `yt-dlp` not installed? YouTube metadata analysis unavailable.")
    except Exception as e:
        print(f"Warning: Failed to initialize YouTube Metadata Tool: {e}")

    print(f"--- Total tools initialized: {len(tools)} ---")
    print(f"Available tool names: {[tool.name for tool in tools]}")
    return tools
