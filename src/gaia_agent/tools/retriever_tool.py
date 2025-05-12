import os
import traceback
from typing import Optional
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from langchain.tools.retriever import (
    create_retriever_tool as lang_create_retriever_tool,
)

# Imports for Supabase and Embeddings
try:
    from supabase.client import Client, create_client
    from langchain_supabase import SupabaseVectorStore

    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
try:
    from langchain_huggingface import HuggingFaceEmbeddings

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

from gaia_agent.logger_config import logger
from gaia_agent.config_loader import get_config_value


# --- Input Schema ---
class RetrieverInput(BaseModel):
    query: str = Field(
        description="The query or question to search for in the document/question database."
    )


# --- Tool Setup Function ---
def setup_retriever_tool() -> Optional[BaseTool]:
    """Sets up the Supabase Retriever Tool if configured and dependencies are available."""
    logger.info("Attempting to set up Supabase Retriever Tool...")

    if not SUPABASE_AVAILABLE:
        logger.warning(
            "Supabase libraries not found (`supabase-py`, `langchain-supabase`). Skipping Retriever tool."
        )
        return None
    if not EMBEDDINGS_AVAILABLE:
        logger.warning(
            "HuggingFaceEmbeddings not found (`langchain-huggingface`, `sentence-transformers`). Skipping Retriever tool."
        )
        return None

    # Get config/env vars
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    table_name = get_config_value(["supabase", "table_name"], "documents")
    query_name = get_config_value(
        ["supabase", "query_name"], "match_documents_langchain"
    )
    embedding_model_name = get_config_value(
        ["huggingface", "embedding_model"], "sentence-transformers/all-mpnet-base-v2"
    )

    if not supabase_url or not supabase_key:
        logger.warning(
            "SUPABASE_URL or SUPABASE_SERVICE_KEY environment variables not set. Skipping Retriever tool."
        )
        return None

    logger.info(
        f"Retriever Config: URL set, Key set, Table='{table_name}', Query='{query_name}', Embedding='{embedding_model_name}'"
    )

    try:
        # Initialize Supabase client
        supabase_client: Client = create_client(supabase_url, supabase_key)
        logger.debug("Supabase client created.")

        # Initialize Embeddings
        logger.debug(f"Initializing embeddings model: {embedding_model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        logger.debug("Embeddings model initialized.")

        # Initialize Vector Store
        logger.debug("Initializing SupabaseVectorStore...")
        vector_store = SupabaseVectorStore(
            client=supabase_client,
            embedding=embeddings,
            table_name=table_name,
            query_name=query_name,
        )
        logger.debug("SupabaseVectorStore initialized.")

        # Create Retriever Tool
        retriever = vector_store.as_retriever()
        logger.debug("Retriever created.")

        retriever_tool = lang_create_retriever_tool(
            retriever=retriever,
            name="question_retriever",  # Changed name to be more specific
            description="Searches a knowledge base (vector store) for documents or previous questions relevant to the input query.",
        )
        logger.info("✅ Supabase Retriever Tool initialized successfully.")
        return retriever_tool

    except Exception as e:
        logger.exception(f"Error setting up Supabase Retriever Tool: {e}")
        return None


# --- Instantiate the tool (or None if setup fails) ---
# This runs when the module is imported
RETRIEVER_TOOL_INSTANCE = setup_retriever_tool()

if __name__ == "__main__":
    # Test the tool setup
    if RETRIEVER_TOOL_INSTANCE:
        logger.info("Retriever Tool is set up and ready to use.")
    else:
        logger.warning("Retriever Tool setup failed. Check logs for details.")

    # Example usage of the tool
    if RETRIEVER_TOOL_INSTANCE:
        test_query = "What is the capital of France?"
        try:
            result = RETRIEVER_TOOL_INSTANCE.run(test_query)
            logger.info(f"Test query result: {result}")
        except Exception as e:
            logger.error(f"Error running test query: {e}")
            logger.debug(traceback.format_exc())
            # Log the full traceback for debugging
            logger.debug("Full traceback:", exc_info=True)
