from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import the loaded configuration dictionary
from config_loader import CONFIG, get_config_value


def get_planner_prompt() -> ChatPromptTemplate:
    """
    Creates the ChatPromptTemplate for the main planner agent.
    Loads the system prompt template from the YAML config.
    """
    # Access the prompt template using dictionary keys or the helper
    # system_prompt_template = CONFIG.get('prompts', {}).get('planner_system', "Default prompt if not found")
    system_prompt_template = get_config_value(
        ["prompts", "planner_system"], "ERROR: Planner prompt not found in config.yaml"
    )

    if "ERROR" in system_prompt_template:
        print(f"Warning: {system_prompt_template}")

    # Note: {tool_descriptions} is filled at runtime in agent_core.py
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_template),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )


if __name__ == "__main__":
    # Example usage of the planner prompt
    planner_prompt = get_planner_prompt()
    print("Planner Prompt Template:", planner_prompt)
    print("Prompt Messages:", planner_prompt.messages)
