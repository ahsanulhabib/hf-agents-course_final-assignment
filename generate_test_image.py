import os
from io import BytesIO
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from lorem_text import lorem
from PIL import Image
import requests

# Hugging Face Inference API endpoint
HF_API_URL = "https://api-inference.huggingface.co/models/{model_id}"


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
        HF_API_URL.format(model_id=model_id), headers=headers, json=payload
    )

    if response.status_code != 200:
        raise Exception(
            f"Failed to generate image: {response.status_code}, {response.text}"
        )

    image_bytes = response.content
    return Image.open(BytesIO(image_bytes))


# Example Usage:
if __name__ == "__main__":
    load_dotenv()
    TEST_DATA_DIR = Path(__file__).parent / "src" / "gaia_test" / "data"
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FILEPATH = TEST_DATA_DIR / "sample_ocr_image.png"
    FILEPATH.touch(exist_ok=True)
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    MODEL_ID = "black-forest-labs/FLUX.1-schnell"

    prompt = f"A clear document page with large black text on white background saying:\n '{lorem.paragraphs(3)}'"
    print(
        "Generating image... This may take some time depending on the model and API queue."
    )

    try:
        generated_image = generate_image_from_prompt(prompt, MODEL_ID, HF_TOKEN)
        generated_image.save(FILEPATH)
        generated_image.show()
    except Exception as e:
        print(f"Error: {e}")
