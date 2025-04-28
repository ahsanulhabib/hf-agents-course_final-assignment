import os
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Import configuration
import config

load_dotenv()


def get_gemini_llm(
    model_name: str = config.GEMINI_MODEL_ID,
    temperature: float = config.DEFAULT_LLM_TEMPERATURE,
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """Initializes and returns a Gemini Chat Model instance."""
    key = api_key or os.getenv("GOOGLE_API_TOKEN")
    if not key:
        raise ValueError("Google API Key not found. Set GOOGLE_API_TOKEN env var.")

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    print(f"Initializing Gemini model: {model_name} with safety settings.")
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=key,
        temperature=temperature,
        safety_settings=safety_settings,
        convert_system_message_to_human=True,
    )


def get_groq_llm(
    model_name: str = config.GROQ_MODEL_ID,
    temperature: float = config.DEFAULT_LLM_TEMPERATURE,
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """Initializes and returns a Groq Chat Model instance."""
    key = api_key or os.getenv("GROQ_API_TOKEN")
    if not key:
        raise ValueError("Groq API Key not found. Set GROQ_API_TOKEN env var.")
    print(f"Initializing Groq model: {model_name}")
    return ChatGroq(model_name=model_name, temperature=temperature, groq_api_key=key)


def get_hf_inference_llm(
    repo_id: str = config.HF_INFERENCE_MODEL_ID,
    temperature: float = config.DEFAULT_LLM_TEMPERATURE,
    max_new_tokens: int = 1024,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> BaseChatModel:
    """Initializes and returns a Hugging Face Inference Endpoint LLM instance."""
    key = api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not key:
        raise ValueError(
            "HF API Token not found. Set HUGGINGFACEHUB_API_TOKEN env var."
        )
    print(f"Initializing HF Inference Endpoint: {repo_id}")
    hf_temp = max(temperature, 0.01) if temperature <= 0 else temperature
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=hf_temp,
        max_new_tokens=max_new_tokens,
        huggingfacehub_api_key=key,
        endpoint_url=api_url,
    )


def get_llm(llm_type: str = config.DEFAULT_PLANNER_LLM) -> BaseChatModel:
    """Gets the specified LLM type based on config."""
    if llm_type == "gemini":
        return get_gemini_llm()
    elif llm_type == "groq":
        return get_groq_llm()
    elif llm_type == "hf":
        return get_hf_inference_llm()
    else:
        print(f"Warning: Unknown LLM type '{llm_type}'. Defaulting to Gemini.")
        return get_gemini_llm()


# --- Example Usage ---
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
