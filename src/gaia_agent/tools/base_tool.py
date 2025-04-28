"""Optional base class if common logic is needed"""

from langchain_core.tools import BaseTool
from pydantic import Field  # For args_schema if needed

# class GaiaBaseTool(BaseTool):
#     # Add common properties or methods here if needed in the future
#     pass
