import os
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint  # Correct import
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()


def get_gemini_llm(
    model_name: str = "gemini-2.5-pro-latest",
    temperature: float = 0.1,
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """Initializes and returns a Gemini Chat Model instance with safety settings."""
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Google API Key not found. Set GOOGLE_API_KEY env var.")

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    print(f"Initializing Gemini model: {model_name} with safety settings.")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=key,
        temperature=temperature,
        safety_settings=safety_settings,
        convert_system_message_to_human=True,  # Often improves compatibility
    )


def get_groq_llm(
    model_name: str = "llama3-8b-8192",
    temperature: float = 0.1,
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """Initializes and returns a Groq Chat Model instance."""
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Groq API Key not found. Set GROQ_API_KEY env var.")
    print(f"Initializing Groq model: {model_name}")
    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
        groq_api_key=key,
    )


def get_hf_inference_llm(
    repo_id: str = "mistralai/Mistral-7B-Instruct-v0.1",  # Example model
    temperature: float = 0.1,
    max_new_tokens: int = 1024,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,  # Optional: for dedicated endpoints
) -> BaseChatModel:
    """
    Initializes and returns a Hugging Face Inference Endpoint LLM instance.
    Uses langchain_huggingface integration.
    """
    key = api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not key:
        raise ValueError(
            "Hugging Face API Token not found. Set HUGGINGFACEHUB_API_TOKEN env var."
        )

    print(f"Initializing Hugging Face Inference Endpoint for model: {repo_id}")
    # Ensure correct parameters are passed based on HuggingFaceEndpoint documentation
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=temperature
        if temperature > 0
        else 0.01,  # Temp must be > 0 for HF Endpoint
        max_new_tokens=max_new_tokens,
        huggingfacehub_api_token=key,
        endpoint_url=api_url,  # Pass None if using standard inference API
        # Add other parameters like top_k, top_p as needed:
        # top_k=50,
        # top_p=0.95,
    )


# --- Example Usage (Optional) ---
if __name__ == "__main__":
    try:
        print("Attempting to initialize LLMs...")
        gemini_llm = get_gemini_llm()
        print(f"Gemini LLM ({gemini_llm.model}) initialized.")

        # Uncomment to test if keys are set in .env
        # if os.getenv("GROQ_API_KEY"):
        #     groq_llm = get_groq_llm()
        #     print(f"Groq LLM ({groq_llm.model_name}) initialized.")
        # else:
        #     print("Skipping Groq LLM init (GROQ_API_KEY not set).")

        # if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        #     hf_llm = get_hf_inference_llm()
        #     print(f"HF Inference LLM ({hf_llm.repo_id}) initialized.")
        # else:
        #      print("Skipping HF Inference LLM init (HUGGINGFACEHUB_API_TOKEN not set).")

    except ValueError as e:
        print(f"Configuration Error initializing LLMs: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during LLM init: {e}")
