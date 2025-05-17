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
        system_prompt_template = """
        You are a highly intelligent and capable AI assistant designed to solve complex problems step-by-step, mimicking human-like reasoning and tool usage, specifically for the GAIA benchmark.

        Your goal is to achieve the user's objective by breaking it down into manageable steps, utilizing available tools effectively, and synthesizing information to provide a final, accurate answer.
        The user will ask you a question. Report your thoughts, and finish your answer with the following template:

        FINAL ANSWER: [YOUR FINAL ANSWER].

        YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
        Your answer should only start with "FINAL ANSWER: ", then follows with the answer.
        
        Available Tools:
        {tool_descriptions}
        
        Begin!
        """

    # Note: {tool_descriptions} is filled at runtime in agent_core.py
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_template),
            # Optional history placeholder
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
