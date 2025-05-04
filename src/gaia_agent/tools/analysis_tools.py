import base64
import io
import mimetypes
import os
import traceback
from pathlib import Path
from typing import Optional, Type, Union

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from PIL import Image

from gaia_agent.llm_config import get_gemini_llm
from gaia_agent.config_loader import get_config_value
from dotenv import load_dotenv

load_dotenv()


# --- Helper Function ---
def _encode_image_to_base64(image_path: str) -> str:
    """Reads an image file and encodes it to a base64 data URI."""
    if Image is None:
        return "Error: Pillow library is not installed. Run `pip install Pillow`."
    try:
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"

        with Image.open(image_path) as img:
            # Determine format for data URI
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith("image"):
                # Fallback or raise error if format unknown/not image
                img_format = img.format if img.format else "png"  # Default to png
                mime_type = f"image/{img_format.lower()}"
                print(
                    f"Warning: Could not guess mime type for {image_path}, using {mime_type}"
                )

            # Save image to buffer in its original format if possible
            buffer = io.BytesIO()
            # Use original format if available, otherwise default (e.g., PNG)
            save_format = img.format or "PNG"
            # Handle potential issues saving certain formats (like WEBP without library)
            try:
                img.save(buffer, format=save_format)
            except OSError:
                print(
                    f"Warning: Could not save in original format '{save_format}', saving as PNG."
                )
                img.save(buffer, format="PNG")
                mime_type = "image/png"  # Update mime type if saved as PNG

            img_bytes = buffer.getvalue()
            base64_str = base64.b64encode(img_bytes).decode("utf-8")
            return f"data:{mime_type};base64,{base64_str}"
    except FileNotFoundError:
        return f"Error: Image file not found at {image_path}"
    except Exception as e:
        return f"Error encoding image {image_path}: {e}\n{traceback.format_exc()}"


# --- Input Schemas ---
class AnalyzeTextInput(BaseModel):
    file_path: str = Field(
        description="The local path to the text file (e.g., /tmp/document.txt)."
    )
    query: Optional[str] = Field(
        None,
        description="Optional specific question about the text content (e.g., 'summarize'). Currently provides basic stats.",
    )


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
class AnalyzeTextTool(BaseTool):
    name: str = "analyze_text_file"
    description: str = (
        "Analyzes a text file (.txt). Provides basic info like line count, word count, "
        "and a preview of the content. Can optionally answer simple questions about content (future)."
    )
    args_schema: Type[BaseModel] = AnalyzeTextInput

    def _run(self, file_path: str, query: Optional[str] = None) -> str:
        try:
            if not os.path.exists(file_path):
                return f"Error: Text file not found at path: {file_path}"

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.splitlines()
            line_count = len(lines)
            word_count = len(content.split())
            char_count = len(content)

            # Generate a preview (e.g., first 5 lines or 500 chars)
            preview_lines = "\n".join(lines[:5])
            if len(preview_lines) > 500:
                preview = content[:500] + "..."
            else:
                preview = preview_lines + ("..." if line_count > 5 else "")

            result = (
                f"Text file '{os.path.basename(file_path)}' analysis:\n"
                f"- Lines: {line_count}\n"
                f"- Words: {word_count}\n"
                f"- Characters: {char_count}\n\n"
                f"Content Preview:\n---\n{preview}\n---\n"
            )

            if query:
                result += f"\nQuery received: '{query}' (Note: Advanced query answering not yet implemented for text files, providing basic stats.)"

            return result

        except Exception as e:
            return f"Error analyzing text file '{os.path.basename(file_path)}': {str(e)}\n{traceback.format_exc()}"


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
            buffer = io.StringIO()
            df.info(buf=buffer)
            result += "Info:\n" + buffer.getvalue() + "\n\n"
            result += "Stats:\n" + df.describe(include="all").to_string() + "\n\n"
            result += (
                "Use Python REPL tool for more complex analysis using path: "
                + file_path
            )
            return result
        except ImportError:
            return "Error: pandas library not installed. Run `pip install pandas`."
        except Exception as e:
            return f"Error analyzing CSV file '{os.path.basename(file_path)}': {str(e)}\n{traceback.format_exc()}"


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
            buffer = io.StringIO()
            df.info(buf=buffer)
            result += "Info:\n" + buffer.getvalue() + "\n\n"
            result += "Stats:\n" + df.describe(include="all").to_string() + "\n\n"
            result += f"Use Python REPL tool for more complex analysis using path: {file_path} and sheet_name='{sheet_name}'"
            return result
        except ImportError:
            return "Error: pandas and openpyxl not installed. Run `pip install pandas openpyxl`."
        except Exception as e:
            return f"Error analyzing Excel file '{os.path.basename(file_path)}': {str(e)}\n{traceback.format_exc()}"


class ExtractImageTextTool(BaseTool):
    name: str = "extract_text_from_image"
    description: str = (
        "Extracts text content from an image file using a powerful multimodal LLM (Groq Llama 3 70B). "
        "Requires the image file to exist locally and the GROQ_API_KEY environment variable to be set."
    )
    args_schema: Type[BaseModel] = ExtractImageTextInput

    def _run(self, image_path: str) -> str:
        print(f"--- Tool: ExtractImageTextTool (LLM OCR) for: {image_path} ---")

        # Check path exists
        if not os.path.exists(image_path):
            return f"Error: Image file not found: {image_path}"

        # Encode image to base64 data URI
        try:
            image_b64_data_uri = _encode_image_to_base64(image_path)
            if image_b64_data_uri.startswith("Error:"):
                return image_b64_data_uri  # Propagate encoding error
        except ImportError:
            # Pillow might be missing if checked earlier but failed import here
            return "Error: Pillow library not installed. Run `pip install Pillow`."
        except Exception as e:
            return f"Error reading/encoding image '{os.path.basename(image_path)}': {e}\n{traceback.format_exc()}"

        # Get Google API Key
        google_api_key = os.getenv("GOOGLE_API_TOKEN") or os.getenv("GOOGLE_API_KEY")

        # Access values using the helper for safety
        model_temp = get_config_value(["llm", "default_temperature"], 0.1)
        model_id = get_config_value(
            ["llm", "models", "gemini"], "gemini-2.5-pro-latest"
        )

        if not google_api_key:
            return "Error: GOOGLE_API_KEY environment variable not set for Gemini OCR."

        # Instantiate Gemini 2.5 Pro
        try:
            llm = get_gemini_llm()
            print(f"Gemini model initialized for OCR: {model_id}")
        except Exception as e:
            return f"Error initializing Gemini model for OCR: {e}"

        # Create Prompt Message for Multimodal Input
        ocr_prompt_text = (
            "Perform OCR on the following image. Extract all text accurately. "
            "Respond ONLY with the extracted text, without any introductory phrases, "
            "explanations, or formatting like ```."
            "Return only the extracted text, no explanations."
        )

        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": ocr_prompt_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_b64_data_uri
                            # f"data:image/png;base64,{image_base64}"
                        },
                    },
                ]
            )
        ]

        # Invoke LLM for OCR
        try:
            print(
                f"Sending image '{os.path.basename(image_path)}' to {model_id} for OCR..."
            )

            response = llm.invoke(message)
            extracted_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            extracted_text = (
                " ".join(str(item) for item in extracted_text)
                if isinstance(extracted_text, list)
                else str(extracted_text)
            )

            print(f"LLM OCR Raw Response: {extracted_text.strip()}")

            if not extracted_text or not extracted_text.strip():
                return f"Processed image '{os.path.basename(image_path)}', but the LLM OCR did not return any text."

            # Return the extracted text, potentially adding context
            return f"Extracted text from '{os.path.basename(image_path)}' (using LLM OCR):\n---\n{extracted_text.strip()}\n---"

        except Exception as e:
            print(
                f"Error during LLM OCR call for '{os.path.basename(image_path)}': {e}"
            )
            return f"Error during LLM OCR call: {e}\n{traceback.format_exc()}"


if __name__ == "__main__":
    # Load test files from data directory
    test_data_dir = Path(__file__).parent.parent.parent / "gaia_test" / "data"
    if not test_data_dir.exists():
        raise FileNotFoundError(
            f"Test data directory not found: {test_data_dir}. Please ensure the test data files are present."
        )
    example_text_path = test_data_dir / "sample_text_file.txt"
    example_csv_path = test_data_dir / "sample_csv_file.csv"
    example_excel_path = test_data_dir / "sample_xlsx_file.xlsx"
    example_image_path = test_data_dir / "sample_ocr_image.png"

    # csv_tool = AnalyzeCsvTool()

    # csv_result = csv_tool._run(
    #     file_path=example_csv_path,
    #     query="What is the average value in the 'age' column?",
    # )

    # print("CSV Analysis Result:\n", csv_result)

    # # Example usage for AnalyzeExcelTool
    # excel_tool = AnalyzeExcelTool()
    # excel_result = excel_tool._run(
    #     file_path=example_excel_path,
    #     query="Who is the next of kin of 'Betty Jones'?",
    #     sheet_name=0,  # or use sheet_name="Sheet1"
    # )
    # print("Excel Analysis Result:\n", excel_result)

    # # Example usage for ExtractImageTextTool
    # image_tool = ExtractImageTextTool()
    # image_result = image_tool._run(image_path=example_image_path)
    # print("Extracted Image Text:\n", image_result)

    project_root = Path(__file__).parent.parent.parent.parent
    test_data_path = project_root / "src" / "gaia_test" / "data"
    print(f"Looking for test data in: {test_data_path}")

    if not test_data_path.is_dir():
        print(f"ERROR: Test data directory not found at '{test_data_path}'.")
        print(
            "Please create it and run 'python generate_test_data.py' from the project root."
        )
        exit(1)

    # --- Test Functions ---
    def run_test(tool_instance: BaseTool, test_name: str, **kwargs):
        print(f"\n--- Testing: {test_name} ({tool_instance.name}) ---")
        print(f"Input Args: {kwargs}")
        try:
            result = tool_instance.run(**kwargs)  # Unpack args for pydantic validation
            print(
                f"Output:\n{result[:1000]}{'...' if len(result) > 1000 else ''}"
            )  # Truncate long output
            if "Error:" in result:
                print("Status: ❌ Failed (Returned Error)")
            else:
                print("Status: ✅ Passed")
        except Exception as e:
            print("Status: ❌ Failed (Threw Exception)")
            print(f"Exception: {e}")
            traceback.print_exc()
        print("-" * (len(test_name) + 14))

    # --- Text Tool Test ---
    if example_text_path.exists():
        text_tool = AnalyzeTextTool()
        run_test(
            text_tool,
            "Text File Analysis",
            file_path=str(example_text_path),
            query="Count lines",
        )
    else:
        print(
            f"\n--- Skipping Text Test: File not found '{str(example_text_path)}' ---"
        )

    # --- CSV Tool Test ---
    if example_csv_path.exists():
        csv_tool = AnalyzeCsvTool()
        run_test(
            csv_tool,
            "CSV Analysis",
            file_path=str(example_csv_path),
            query="Get basic stats",
        )
    else:
        print(f"\n--- Skipping CSV Test: File not found '{str(example_csv_path)}' ---")

    # --- Excel Tool Test ---
    if example_excel_path.exists():
        excel_tool = AnalyzeExcelTool()
        run_test(
            excel_tool,
            "Excel Analysis",
            file_path=str(example_excel_path),
            query="Get basic stats",
            sheet_name=0,
        )
    else:
        print(
            f"\n--- Skipping Excel Test: File not found '{str(example_excel_path)}' ---"
        )
        print("   (Run 'python generate_test_data.py' with pandas/openpyxl installed)")

    # --- Image OCR (LLM) Tool Test ---
    if example_image_path.exists():
        if os.getenv("GOOGLE_API_TOKEN") or os.getenv("GOOGLE_API_KEY"):
            ocr_tool = ExtractImageTextTool()
            run_test(ocr_tool, "Image OCR (LLM)", image_path=str(example_image_path))
        else:
            print("\n--- Skipping OCR (LLM) Test: GOOGLE_API_TOKEN not set in .env ---")
    else:
        print(
            f"\n--- Skipping OCR (LLM) Test: File not found '{str(example_image_path)}' ---"
        )
        print("   (Place a sample PNG image with text in 'test_data/')")

    print("\n--- Analysis Tools Tests Finished ---")
