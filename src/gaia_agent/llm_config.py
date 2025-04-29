import os
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Import the loaded configuration dictionary and helper
from gaia_agent.config_loader import CONFIG, get_config_value

load_dotenv()

# Access values using the helper for safety
_DEFAULT_LLM_TEMP = get_config_value(["llm", "default_temperature"], 0.1)
_GEMINI_MODEL_ID = get_config_value(
    ["llm", "models", "gemini"], "gemini-2.5-pro-exp-03-25"
)
_GROQ_MODEL_ID = get_config_value(["llm", "models", "groq"], "llama-3.3-70b-versatile")
_HF_MODEL_ID = get_config_value(
    ["llm", "models", "hf_inference"], "google/gemma-3-27b-it"
)
_DEFAULT_PLANNER = get_config_value(["llm", "default_planner"], "gemini")


def get_gemini_llm(
    model_name: str = _GEMINI_MODEL_ID,
    temperature: float = _DEFAULT_LLM_TEMP,
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """
    Initializes and returns a Gemini Chat Model instance with specified configuration.

    This function creates an instance of the Gemini chat model (ChatGoogleGenerativeAI) using the provided
    model name, temperature, and API key. It also applies predefined safety settings to filter out
    harmful content such as harassment, hate speech, sexually explicit material, and dangerous content.

    Args:
        model_name (str): The identifier of the Gemini model to use. Defaults to _GEMINI_MODEL_ID.
        temperature (float): The sampling temperature for response generation, controlling randomness.
            Defaults to _DEFAULT_LLM_TEMP.
        api_key (Optional[str]): Google API key for authentication. If not provided, the function
            attempts to retrieve it from the 'GOOGLE_API_KEY' environment variable.

    Returns:
        BaseChatModel: An instance of ChatGoogleGenerativeAI configured with the specified parameters
            and safety settings.

    Raises:
        ValueError: If the API key is not provided and cannot be found in the environment variables.

    Notes:
        - The function enforces medium and above blocking thresholds for various harm categories.
        - The system message is converted to a human message for compatibility with the model.
    """
    key = api_key or os.getenv("GOOGLE_API_TOKEN") or os.getenv("GOOGLE_API_KEY")

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
        # convert_system_message_to_human=True,
        max_output_tokens=65536,
    )


def get_groq_llm(
    model_name: str = _GROQ_MODEL_ID,
    temperature: float = _DEFAULT_LLM_TEMP,
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """
    Initializes and returns an instance of the Groq Chat Model.

    This function sets up a Groq-based large language model (LLM) for chat-based interactions.
    It retrieves the API key from the provided argument or from the environment variables
    `GROQ_API_TOKEN` or `GROQ_API_KEY`. If no API key is found, it raises a ValueError.
    The function also allows customization of the model name and temperature.

    Args:
        model_name (str, optional): The identifier for the Groq model to use.
            Defaults to the value of `_GROQ_MODEL_ID`.
        temperature (float, optional): Sampling temperature to use for response generation.
            Higher values make output more random, lower values make it more deterministic.
            Defaults to `_DEFAULT_LLM_TEMP`.
        api_key (Optional[str], optional): Groq API key. If not provided, the function
            attempts to retrieve it from environment variables.

    Returns:
        BaseChatModel: An initialized instance of the Groq Chat Model ready for use.

    Raises:
        ValueError: If no API key is provided or found in the environment variables.

    Side Effects:
        Prints a message indicating the initialization of the Groq model with the specified name.
    """
    key = api_key or os.getenv("GROQ_API_TOKEN") or os.getenv("GROQ_API_KEY")

    if not key:
        raise ValueError("Groq API Key not found. Set GROQ_API_KEY env var.")

    print(f"Initializing Groq model: {model_name}")

    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
        groq_api_key=key,
        max_tokens=65536,
    )


def get_hf_inference_llm(
    repo_id: str = _HF_MODEL_ID,
    temperature: float = _DEFAULT_LLM_TEMP,
    max_new_tokens: int = 1024,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> BaseChatModel:
    """
    Initializes and returns a Hugging Face Inference Endpoint LLM instance.

    This function creates an instance of a language model using the Hugging Face Inference Endpoint.
    It allows customization of the model repository, temperature, maximum number of new tokens, API key, and endpoint URL.

    Args:
        repo_id (str): The Hugging Face model repository ID to use. Defaults to the value of _HF_MODEL_ID.
        temperature (float): Sampling temperature for the model's output. Higher values increase randomness.
            If a value less than or equal to 0 is provided, it is set to a minimum of 0.01. Defaults to _DEFAULT_LLM_TEMP.
        max_new_tokens (int): The maximum number of new tokens to generate in the response. Defaults to 1024.
        api_key (Optional[str]): Hugging Face API token. If not provided, the function attempts to read it from
            the 'HUGGINGFACEHUB_API_TOKEN' environment variable.
        api_url (Optional[str]): Optional custom endpoint URL for the Hugging Face Inference API.

    Returns:
        BaseChatModel: An instance of the HuggingFaceEndpoint class configured with the specified parameters.

    Raises:
        ValueError: If the API key is not provided and cannot be found in the environment variables.
        ValueError: If the API URL is invalid or not reachable.

    Notes:
        - Prints a message indicating which Hugging Face Inference Endpoint is being initialized.
        - Ensures the temperature is never set below 0.01 to avoid invalid model behavior.
    """
    key = (
        api_key
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_KEY")
    )

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
        huggingfacehub_api_token=key,
        endpoint_url=api_url,
    )


def get_llm(llm_type: str = _DEFAULT_PLANNER) -> BaseChatModel:
    """
    Retrieves an instance of a language model (LLM) based on the specified type.

    Args:
        llm_type (str, optional): The type of language model to retrieve.
            Supported values are:
                - "gemini": Returns an instance of the Gemini LLM.
                - "groq": Returns an instance of the Groq LLM.
                - "hf": Returns an instance of the Hugging Face Inference LLM.
            If an unknown value is provided, the function defaults to returning the Gemini LLM.
            The default value is set by the module-level constant `_DEFAULT_PLANNER`.

    Returns:
        BaseChatModel: An instance of the selected language model class.

    Side Effects:
        Prints diagnostic messages indicating the requested LLM type and warnings for unknown types.

    Raises:
        None

    Note:
        The actual LLM instance returned depends on the helper functions `get_gemini_llm`, `get_groq_llm`, and `get_hf_inference_llm`,
        which must be defined elsewhere in the codebase.
    """
    print(f"Attempting to get LLM of type: {llm_type}")

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
