import os
from typing import Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.language_models.chat_models import BaseChatModel
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()


def get_gemini_llm(
    model_name: str = "gemini-2.5-pro-latest",
    temperature: float = 0.1,
    API_TOKEN: Optional[str] = None,
) -> BaseChatModel:
    """Initializes and returns a Gemini Chat Model instance with safety settings."""
    key = API_TOKEN or os.getenv("GOOGLE_API_TOKEN")
    if not key:
        raise ValueError("Google API Key not found. Set GOOGLE_API_TOKEN env var.")

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    print(f"Initializing Gemini model: {model_name} with safety settings.")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_API_TOKEN=key,
        temperature=temperature,
        safety_settings=safety_settings,
        convert_system_message_to_human=True,
    )


def get_groq_llm(
    model_name: str = "gemma2-9b-it",
    temperature: float = 0.1,
    API_TOKEN: Optional[str] = None,
) -> BaseChatModel:
    """Initializes and returns a Groq Chat Model instance."""
    key = API_TOKEN or os.getenv("GROQ_API_TOKEN")
    if not key:
        raise ValueError("Groq API Key not found. Set GROQ_API_TOKEN env var.")
    print(f"Initializing Groq model: {model_name}")
    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
        groq_API_TOKEN=key,
    )


def get_hf_inference_llm(
    repo_id: str = "google/gemma-3-27b-it",  # Example model
    temperature: float = 0.1,
    max_new_tokens: int = 1024,
    API_TOKEN: Optional[str] = None,
    api_url: Optional[str] = None,  # Optional: for dedicated endpoints
) -> BaseChatModel:
    """
    Initializes and returns a Hugging Face Inference Endpoint LLM instance.

    Args:
        repo_id: The repository ID of the model on Hugging Face Hub.
        temperature: The sampling temperature.
        max_new_tokens: Max tokens to generate.
        API_TOKEN: HF API Token. Reads from HUGGINGFACEHUB_API_TOKEN env var if None.
        api_url: Optional URL for a dedicated Inference Endpoint.

    Returns:
        An instance of BaseChatModel (HuggingFaceEndpoint).

    Raises:
        ValueError: If the API key is not found.
    """
    key = API_TOKEN or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not key:
        raise ValueError(
            "Hugging Face API Token not found. Set HUGGINGFACEHUB_API_TOKEN env var."
        )

    print(f"Initializing Hugging Face Inference Endpoint for model: {repo_id}")
    # Ensure temperature is within valid range for HF endpoint (e.g., > 0.0)
    hf_temp = max(temperature, 0.01) if temperature <= 0 else temperature

    return HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=hf_temp,
        max_new_tokens=max_new_tokens,
        huggingfacehub_api_token=key,
        endpoint_url=api_url,  # Pass None if using standard inference API
        # Add other parameters like top_k, top_p as needed:
        # top_k=50,
        # top_p=0.95,
    )


# --- Example Usage (Optional) ---
if __name__ == "__main__":
    print("--- LLM Configuration Test ---")
    try:
        print("\nAttempting to initialize Gemini LLM...")
        if os.getenv("GOOGLE_API_TOKEN"):
            gemini_llm = get_gemini_llm()
            print(f"✅ Gemini LLM ({gemini_llm.model}) initialized.")
        else:
            print("⚠️ GOOGLE_API_TOKEN not set, skipping Gemini.")

        print("\nAttempting to initialize Groq LLM...")
        if os.getenv("GROQ_API_TOKEN"):
            groq_llm = get_groq_llm()
            print(f"✅ Groq LLM ({groq_llm.model_name}) initialized.")
        else:
            print("⚠️ GROQ_API_TOKEN not set, skipping Groq.")

        print("\nAttempting to initialize Hugging Face Inference LLM...")
        if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
            hf_llm = get_hf_inference_llm()  # Uses default Mistral-7B
            print(f"✅ HF Inference LLM ({hf_llm.repo_id}) initialized.")
        else:
            print("⚠️ HUGGINGFACEHUB_API_TOKEN not set, skipping HF Inference.")

    except ValueError as e:
        print(f"❌ Error initializing LLMs: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    print("--- LLM Configuration Test Finished ---")
