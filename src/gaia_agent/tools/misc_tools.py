import os
import io
import json
import time
import contextlib
import yt_dlp
from urllib.parse import urlparse
from typing import Optional, Type
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from langchain_experimental.tools import PythonREPLTool as LangchainPythonREPL


# --- Input Schemas ---
class PythonReplInput(BaseModel):
    command: str = Field(description="The Python command to execute.")


class YoutubeMetadataInput(BaseModel):
    youtube_url: str = Field(description="The full URL of the YouTube video.")


# --- Tool Classes ---
class PythonReplTool(BaseTool):
    name: str = "python_repl"
    description: str = (
        "A Python shell. Use this to execute python commands for calculations, data manipulation, "
        "code execution, or checking file existence/content (e.g., `os.path.exists('/path/to/file')`). "
        "ALWAYS `print()` the results you need to see. Be careful with file system operations."
    )
    args_schema: Type[BaseModel] = PythonReplInput

    def __init__(self):
        super().__init__()
        try:
            # Initialize the underlying Langchain tool
            self._repl_tool = LangchainPythonREPL()
        except ImportError:
            raise ImportError(
                "Python REPL tool requires `lark-parser`. Run `pip install lark-parser`."
            )

    def _run(self, command: str) -> str:
        try:
            # Sanitize input slightly? For now, pass directly.
            # Be aware of security implications if exposing this publicly.
            return self._repl_tool.run(command)
        except Exception as e:
            return f"Error executing Python command: {e}"


class AnalyzeYoutubeMetadataTool(BaseTool):
    name: str = "analyze_youtube_metadata"
    description: str = (
        "Extracts metadata (title, description, uploader etc.) from a YouTube video URL using yt-dlp "
        "and provides a summary based *only* on that metadata. This tool does NOT watch the video content."
    )
    args_schema: Type[BaseModel] = YoutubeMetadataInput

    def _run(self, youtube_url: str) -> str:
        print(f"--- Tool: analyze_youtube_metadata ({youtube_url}) ---")
        try:
            parsed_url = urlparse(youtube_url)
            if not all([parsed_url.scheme, parsed_url.netloc]) or (
                "youtube.com" not in parsed_url.netloc
                and "youtu.be" not in parsed_url.netloc
            ):
                return "Error: Please provide a valid YouTube video URL."

            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": "in_playlist",
                "forcejson": True,
                "skip_download": True,
                "youtube_include_dash_manifest": False,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    video_info = ydl.extract_info(youtube_url, download=False)
                    # Extract and clean title
                    title = video_info.get("title", "N/A")
                    if isinstance(title, str):
                        title = title.strip()

                    # Extract and clean description
                    description = video_info.get("description", "N/A")
                    if isinstance(description, str):
                        description = description.strip()

                    # Extract and clean uploader
                    uploader = video_info.get("uploader", "N/A")
                    if isinstance(uploader, str):
                        uploader = uploader.strip()

                    # Extract and format duration
                    duration_s = video_info.get("duration")
                    duration = (
                        time.strftime("%H:%M:%S", time.gmtime(duration_s))
                        if duration_s
                        else "N/A"
                    )

                    # Extract views
                    views = video_info.get("view_count", "N/A")

                    # Extract and format date
                    date = video_info.get("upload_date", "N/A")
                    if isinstance(date, str):
                        date = date.strip()
                    if date and len(date) == 8:
                        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

                    # Compose summary
                    summary = (
                        f"Metadata for YouTube video: {youtube_url}\n"
                        f"Title: {title}\n"
                        f"Uploader: {uploader}\n"
                        f"Date: {date}\n"
                        f"Duration: {duration}\n"
                        f"Views: {views}\n"
                        f"Description:\n---\n"
                        f"{description}\n"
                        "---\n"
                        "Note: Metadata only."
                    )
                    return summary
                except yt_dlp.utils.DownloadError as e:
                    if (
                        "age restricted" in str(e).lower()
                        or "sign in" in str(e).lower()
                    ):
                        return f"Error: Video {youtube_url} likely age-restricted/requires login."
                    elif "video unavailable" in str(e).lower():
                        return f"Error: Video {youtube_url} unavailable."
                    else:
                        return f"Error: yt-dlp failed for {youtube_url}: {str(e)}"
                except Exception as e:
                    return f"Error: Unexpected metadata extraction error for {youtube_url}: {str(e)}"
        except ImportError:
            return "Error: yt-dlp library not installed. Run `pip install yt-dlp`."
        except Exception as e:
            return f"Error setting up YouTube analysis for {youtube_url}: {str(e)}"


if __name__ == "__main__":
    # Test PythonReplTool
    print("Testing PythonReplTool...")
    python_tool = PythonReplTool()
    result = python_tool._run("print(2 + 2)")
    if "4" in result:
        print("PythonReplTool result: ✅", result)
    else:
        print("PythonReplTool result: ❌", result)

    # Test AnalyzeYoutubeMetadataTool with a valid YouTube URL
    print("\nTesting AnalyzeYoutubeMetadataTool with a valid URL...")
    youtube_tool = AnalyzeYoutubeMetadataTool()
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    metadata_result = youtube_tool._run(test_url)
    if "Metadata for YouTube video" in metadata_result:
        print("AnalyzeYoutubeMetadataTool result: ✅\n", metadata_result)
    else:
        print("AnalyzeYoutubeMetadataTool result: ❌\n", metadata_result)

    # Test AnalyzeYoutubeMetadataTool with an invalid URL
    print("\nTesting AnalyzeYoutubeMetadataTool with an invalid URL...")
    invalid_url = "https://www.example.com/"
    invalid_result = youtube_tool._run(invalid_url)
    if "Error" in invalid_result:
        print("AnalyzeYoutubeMetadataTool invalid URL result: ✅\n", invalid_result)
    else:
        print("AnalyzeYoutubeMetadataTool invalid URL result: ❌\n", invalid_result)
