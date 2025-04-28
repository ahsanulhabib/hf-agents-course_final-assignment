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
    repl_tool: LangchainPythonREPL

    def __init__(self):
        super().__init__()
        try:
            # Initialize the underlying Langchain tool
            self.repl_tool = LangchainPythonREPL()
        except ImportError:
            raise ImportError(
                "Python REPL tool requires `lark-parser`. Run `pip install lark-parser`."
            )

    def _run(self, command: str) -> str:
        try:
            # Sanitize input slightly? For now, pass directly.
            # Be aware of security implications if exposing this publicly.
            return self.repl_tool.run(command)
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
            extracted_info_json = None
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    buffer = io.StringIO()
                    with contextlib.redirect_stdout(buffer):
                        ydl.extract_info(youtube_url, download=False)
                    extracted_info_json = buffer.getvalue()
                    if not extracted_info_json:
                        return f"Error: yt-dlp returned no metadata for {youtube_url}."

                    first_video_info = None
                    for line in extracted_info_json.strip().split("\n"):
                        try:
                            first_video_info = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            continue
                    if not first_video_info:
                        return (
                            f"Error: Could not parse metadata JSON for {youtube_url}."
                        )

                    title = first_video_info.get("title", "N/A")
                    description = first_video_info.get("description", "N/A")
                    uploader = first_video_info.get("uploader", "N/A")
                    duration_s = first_video_info.get("duration")
                    duration = (
                        time.strftime("%H:%M:%S", time.gmtime(duration_s))
                        if duration_s
                        else "N/A"
                    )
                    views = first_video_info.get("view_count", "N/A")
                    date = first_video_info.get("upload_date", "N/A")
                    if date and len(date) == 8:
                        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

                    summary = (
                        f"Metadata for YouTube video: {youtube_url}\nTitle: {title}\nUploader: {uploader}\n"
                        f"Date: {date}\nDuration: {duration}\nViews: {views}\nDesc (truncated):\n---\n"
                        f"{description[:1000]}{'...' if len(description) > 1000 else ''}\n---\nNote: Metadata only."
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
