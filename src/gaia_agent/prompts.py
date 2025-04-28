from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from . import config  # Import from config file


def get_planner_prompt() -> ChatPromptTemplate:
    """
    Creates the ChatPromptTemplate for the main planner agent.
    Dynamically injects tool descriptions into the system prompt template.
    """
    # The system prompt template is now loaded from config
    system_prompt_template = config.PLANNER_SYSTEM_PROMPT_TEMPLATE

    # Note: The {tool_descriptions} placeholder will be filled at runtime
    # in agent_core.py before invoking the LLM.
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_template),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
