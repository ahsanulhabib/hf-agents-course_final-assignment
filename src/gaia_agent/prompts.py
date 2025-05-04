from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from gaia_agent.config_loader import get_config_value


def get_planner_prompt() -> ChatPromptTemplate:
    """
    Creates the ChatPromptTemplate for the main planner agent.
    Loads the system prompt template from the YAML config.
    """
    # Access the prompt template using the helper for safety
    system_prompt_template = get_config_value(
        keys=["prompts", "planner_system"],
        default="ERROR: Planner system prompt not found in config.yaml",
    )

    if "ERROR" in system_prompt_template:
        print(f"Warning: {system_prompt_template}")
        # Provide a minimal fallback prompt
        system_prompt_template = "You are a helpful AI assistant. Use the available tools to answer the user's question.\nAvailable Tools:\n{tool_descriptions}\nBegin!"

    # Note: {tool_descriptions} is filled at runtime in agent_core.py
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_template),
            # Optional history placeholder (if you implement chat memory later)
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            # The user's current input
            ("human", "{input}"),
        ]
    )


if __name__ == "__main__":
    # Example usage of the planner prompt
    planner_prompt = get_planner_prompt()
    print("Planner Prompt Template:", planner_prompt)
    print("Prompt Messages:", planner_prompt.messages)
