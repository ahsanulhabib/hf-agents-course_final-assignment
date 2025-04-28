import os
import tempfile
import requests
import re
import uuid
from urllib.parse import urlparse
from typing import Optional, Type
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool


# --- Input Schemas ---
class SaveContentInput(BaseModel):
    content: str = Field(description="The text content to save.")
    filename: Optional[str] = Field(None, description="Optional desired filename.")


class DownloadFileInput(BaseModel):
    url: str = Field(description="The URL of the file to download.")
    filename: Optional[str] = Field(None, description="Optional desired filename.")


# --- Tool Classes ---
class SaveContentTool(BaseTool):
    name: str = "save_content_to_file"
    description: str = (
        "Saves the given text content to a temporary file and returns the file path. "
        "Use this when you need to make text content available as a file for other tools."
    )
    args_schema: Type[BaseModel] = SaveContentInput

    def _run(self, content: str, filename: Optional[str] = None) -> str:
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


class DownloadFileTool(BaseTool):
    name: str = "download_file_from_url"
    description: str = (
        "Downloads a file from a given URL and saves it to a temporary location. "
        "Handles common file types based on URL or content."
    )
    args_schema: Type[BaseModel] = DownloadFileInput

    def _run(self, url: str, filename: Optional[str] = None) -> str:
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
                                filename = fname[0] if fname else None
                    if not filename:
                        content_type = (
                            requests.head(url, timeout=10)
                            .headers.get("content-type", "")
                            .split(";")[0]
                        )
                        ext_map = {
                            "application/pdf": ".pdf",
                            "image/jpeg": ".jpg",
                            "image/png": ".png",
                            "text/csv": ".csv",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                        }
                        ext = ext_map.get(content_type, ".download")
                        filename = f"downloaded_{uuid.uuid4().hex[:8]}{ext}"
                except Exception as e:
                    print(
                        f"Warning: Could not determine filename from URL {url}: {e}. Generating random name."
                    )
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
