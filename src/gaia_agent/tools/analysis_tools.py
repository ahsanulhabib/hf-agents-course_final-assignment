import os
import io
from typing import Optional, Type, Union
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool


# --- Input Schemas ---
class AnalyzeCsvInput(BaseModel):
    file_path: str = Field(
        description="The local path to the CSV file (e.g., /tmp/data.csv)."
    )
    query: str = Field(
        description="The specific question to answer about the CSV data."
    )


class AnalyzeExcelInput(BaseModel):
    file_path: str = Field(
        description="The local path to the Excel file (e.g., /tmp/data.xlsx)."
    )
    query: str = Field(description="The specific question to answer about the data.")
    sheet_name: Optional[Union[str, int]] = Field(
        0, description="Name or index of the sheet to analyze (default is 0)."
    )


class ExtractImageTextInput(BaseModel):
    image_path: str = Field(
        description="The local file path to the image file (e.g., /tmp/my_image.png)."
    )


# --- Tool Classes ---
class AnalyzeCsvTool(BaseTool):
    name: str = "analyze_csv_file"
    description: str = (
        "Analyzes a CSV file using pandas to answer a specific query about its content. "
        "Provides basic info (rows, columns, headers) and attempts to answer the query."
    )
    args_schema: Type[BaseModel] = AnalyzeCsvInput

    def _run(self, file_path: str, query: str) -> str:
        try:
            import pandas as pd

            if not os.path.exists(file_path):
                return f"Error: CSV file not found at path: {file_path}"

            df = pd.read_csv(file_path)
            num_rows, num_cols = df.shape
            columns = ", ".join(df.columns)
            result = f"CSV '{os.path.basename(file_path)}': {num_rows} rows, {num_cols} cols.\nCols: {columns}\n\nQuery: '{query}'\n\n"
            result += "Head:\n" + df.head().to_string() + "\n\n"
            result += "Stats:\n" + df.describe(include="all").to_string() + "\n\n"
            result += (
                "Use Python REPL tool for more complex analysis using path: "
                + file_path
            )
            return result
        except ImportError:
            return "Error: pandas library not installed. Run `pip install pandas`."
        except Exception as e:
            return f"Error analyzing CSV file '{os.path.basename(file_path)}': {str(e)}"


class AnalyzeExcelTool(BaseTool):
    name: str = "analyze_excel_file"
    description: str = (
        "Analyzes an Excel file (.xls, .xlsx) using pandas to answer a specific query about its content. "
        "Provides basic info for a specified sheet."
    )
    args_schema: Type[BaseModel] = AnalyzeExcelInput

    def _run(
        self, file_path: str, query: str, sheet_name: Optional[Union[str, int]] = 0
    ) -> str:
        try:
            import pandas as pd

            if not os.path.exists(file_path):
                return f"Error: Excel file not found at path: {file_path}"
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            except Exception as read_e:
                try:
                    xls = pd.ExcelFile(file_path)
                    sheets = xls.sheet_names
                    return f"Error reading sheet '{sheet_name}': {read_e}. Available: {sheets}"
                except Exception as inner_e:
                    return f"Error reading Excel file: {read_e} (also failed list sheets: {inner_e})"

            num_rows, num_cols = df.shape
            columns = ", ".join(df.columns)
            result = f"Excel '{os.path.basename(file_path)}', Sheet '{sheet_name}': {num_rows} rows, {num_cols} cols.\nCols: {columns}\n\nQuery: '{query}'\n\n"
            result += "Head:\n" + df.head().to_string() + "\n\n"
            result += "Stats:\n" + df.describe(include="all").to_string() + "\n\n"
            result += f"Use Python REPL tool for more complex analysis using path: {file_path} and sheet_name='{sheet_name}'"
            return result
        except ImportError:
            return "Error: pandas and openpyxl not installed. Run `pip install pandas openpyxl`."
        except Exception as e:
            return (
                f"Error analyzing Excel file '{os.path.basename(file_path)}': {str(e)}"
            )


class ExtractImageTextTool(BaseTool):
    name: str = "extract_text_from_image"
    description: str = (
        "Extracts text content from an image file using Optical Character Recognition (OCR). "
        "Requires the image file to exist locally and Tesseract OCR engine to be installed."
    )
    args_schema: Type[BaseModel] = ExtractImageTextInput

    def _run(self, image_path: str) -> str:
        try:
            from PIL import Image
            import pytesseract

            if not os.path.exists(image_path):
                return f"Error: Image file not found: {image_path}"
            try:
                pytesseract.get_tesseract_version()
            except Exception:
                return "Error: Tesseract OCR engine not found/configured. Install and add to PATH."

            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            if not text.strip():
                return f"Processed image '{os.path.basename(image_path)}', but no text detected."
            else:
                return f"Extracted text from '{os.path.basename(image_path)}':\n---\n{text}\n---"
        except ImportError:
            return (
                "Error: Libraries not installed. Run `pip install Pillow pytesseract`."
            )
        except FileNotFoundError:
            return "Error: Tesseract OCR engine not found/configured."
        except Exception as e:
            if "TesseractNotFoundError" in str(type(e)):
                return "Error: Tesseract OCR engine not found/configured."
            return (
                f"Error extracting text from '{os.path.basename(image_path)}': {str(e)}"
            )
