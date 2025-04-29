import os
import sys
import traceback
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from lorem_text import lorem
from PIL import Image

# --- Configuration ---
TEST_DATA_DIR = Path(__file__).parent / "src" / "gaia_test" / "data"
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
# Define files and their content
FILES_TO_GENERATE = {
    "sample_text_file.txt": """This is a sample text file.
It contains multiple lines.
Useful for testing file saving and reading operations.
Line 4.
End of file.""",
    "sample_csv_file.csv": """PatientID,Name,DateOfBirth,AdmissionDate,Room,Diagnosis,Medication,NextOfKin,Age
1001,John Smith,1940-05-12,2023-01-05,12A,Dementia,Donepezil,Mary Smith,83
1002,Betty Jones,1935-11-23,2023-01-12,15B,Arthritis,Ibuprofen,Peter Jones,88
1003,Alan Brown,1938-07-30,2023-01-15,8C,Diabetes,Metformin,Susan Brown,85
1004,Margaret Lee,1942-03-18,2023-01-20,10D,Hypertension,Amlodipine,David Lee,81
1005,Frank Wilson,1937-09-05,2023-01-25,7A,Heart Failure,Furosemide,Emily Wilson,86
1006,Elsie White,1941-12-01,2023-01-28,9B,Parkinson's,Levodopa,James White,82""",
    # Excel file will be generated using pandas if available
    "sample_xlsx_file.xlsx": None,
    # Placeholder instruction for image file
    "sample_ocr_image.png": None,
}
# --- End Configuration ---


def generate_excel_file(filepath: Path, csv_content: str):
    """Generates an Excel file from CSV content using pandas."""
    print(f"  Attempting to generate Excel file: {filepath}...")
    try:
        import pandas as pd
        import io

        # Check for openpyxl which is needed for .xlsx writing
        try:
            import openpyxl
        except ImportError:
            print("  ERROR: `openpyxl` library not found. Cannot generate Excel file.")
            print("  Install it using: pip install openpyxl")
            return False

        # Read CSV data into pandas DataFrame
        csv_file_like = io.StringIO(csv_content)
        df = pd.read_csv(csv_file_like)

        # Write DataFrame to Excel file
        df.to_excel(filepath, index=False, engine="openpyxl")
        print(f"✅ Successfully generated Excel file: {filepath}")
        return True

    except ImportError:
        print("  ERROR: `pandas` library not found. Cannot generate Excel file.")
        print("  Install it using: pip install pandas")
        return False
    except Exception as e:
        print(f"  ERROR generating Excel file '{filepath}': {e}")
        traceback.print_exc()
        return False


def generate_image_from_prompt(
    prompt: str,
    model_id: str,
    hf_token: str,
    guidance_scale: Optional[float] = 7.5,
    num_inference_steps: Optional[int] = 50,
) -> Image.Image:
    """
    Generate an image from a text prompt using the Hugging Face Inference API.

    Args:
        prompt (str): The text prompt for image generation.
        model_id (str): The model ID from Hugging Face (e.g., "stabilityai/stable-diffusion-2").
        hf_token (str): Your Hugging Face API token.
        guidance_scale (float, optional): Classifier-free guidance scale. Defaults to 7.5.
        num_inference_steps (int, optional): Number of denoising steps. Defaults to 50.

    Returns:
        Image.Image: The generated image.
    """
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Accept": "application/json",
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
        },
    }

    response = requests.post(
        "https://api-inference.huggingface.co/models/{model_id}".format(
            model_id=model_id
        ),
        headers=headers,
        json=payload,
    )

    if response.status_code != 200:
        raise Exception(
            f"Failed to generate image: {response.status_code}, {response.text}"
        )

    image_bytes = response.content
    return Image.open(BytesIO(image_bytes))


def create_test_files():
    """Creates the test_data directory and populates it with sample files."""
    print(f"Ensuring test data directory exists: '{TEST_DATA_DIR}'")

    print("\nGenerating test files:")
    for filename, content in FILES_TO_GENERATE.items():
        filepath = TEST_DATA_DIR / filename
        print(f"- Processing: {filename}")

        if filename.endswith(".xlsx"):
            # Generate Excel from CSV content
            csv_filename = filename.replace("xlsx", "csv")
            if csv_filename in FILES_TO_GENERATE:
                generate_excel_file(filepath, FILES_TO_GENERATE[csv_filename])
            else:
                print(
                    f"  Skipping Excel generation: Corresponding CSV '{csv_filename}' not defined."
                )
        elif (
            filename.endswith(".png")
            or filename.endswith(".jpg")
            or filename.endswith(".jpeg")
        ):
            # Provide instructions for image files
            if not filepath.exists():
                try:
                    load_dotenv()  # Load environment variables from .env file
                    # Check if Hugging Face API token is set
                    hf_token = os.environ.get(
                        "HUGGINGFACEHUB_API_TOKEN"
                    ) or os.environ.get("HUGGINGFACEHUB_API_KEY")
                    model_id = "black-forest-labs/FLUX.1-schnell"
                    if not hf_token:
                        print(
                            "  ERROR: HUGGINGFACEHUB_API_TOKEN not found in .env or environment."
                        )
                        print(
                            "  ACTION REQUIRED: Please set the token in your environment or .env file."
                        )
                        continue

                    print(f"  Attempting to generate image file: {filepath}...")
                    # Generate a prompt for the image
                    prompt = f"A clear document page with large black text on white background saying:\n '{lorem.paragraphs(3)}'"
                    print(
                        "Generating image... This may take some time depending on the model and API queue."
                    )

                    generated_image = generate_image_from_prompt(
                        prompt, model_id, hf_token
                    )
                    generated_image.save(filepath)
                    # generated_image.show()  # Show the generated image

                except Exception as e:
                    print(f"  ERROR: Could not generate image automatically: {str(e)}")
                    print(
                        f"  ACTION REQUIRED: Please place a sample image file named '{filename}'"
                    )
                    print(f"                 in the '{TEST_DATA_DIR}' directory.")
                    print(
                        f"                 For '{'ocr' if 'ocr' in filename else 'image'}' testing, ensure it contains some text."
                    )
            else:
                print(f"  Skipping: '{filename}' already exists.")
        elif content is not None:
            # Write text-based files (txt, csv)
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Successfully created/updated: {filepath}")
            except OSError as e:
                print(f"  ERROR creating file '{filepath}': {e}")
        else:
            print(
                f"  Skipping: No content defined for '{filename}' (handled separately)."
            )

    print(f"\nTest data generation process complete for '{TEST_DATA_DIR}'.")
    print("Remember to add any required image files manually if prompted.")


if __name__ == "__main__":
    print("--- Test Data Generator ---")
    # Ensure the script is run from the project root where test_data should be created
    if not Path("pyproject.toml").exists() and not Path("src").exists():
        print(
            "ERROR: Please run this script from the root directory of the 'gaia-agent-hf' project."
        )
        sys.exit(1)

    print("=" * 50)
    create_test_files()
    print("=" * 50)
    print("---------------------------")
