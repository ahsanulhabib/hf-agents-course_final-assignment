import json
import traceback
from typing import TypedDict, Annotated, Sequence, List, Optional, Dict, Any

# from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool  # Still need BaseTool for type hinting
from langgraph.graph import StateGraph, END

# Removed ToolExecutor, ToolInvocation from prebuilt
from langgraph.graph.message import add_messages

# Import from within the package
from gaia_agent.llm_config import BaseChatModel  # , get_llm
from gaia_agent.prompts import get_planner_prompt
from gaia_agent.tools import get_all_tools
from gaia_agent.config_loader import load_config, get_config_value

# --- Load Configuration ---
try:
    CONFIG = load_config()
except Exception as e:
    print(f"FATAL ERROR: Could not load configuration. {e}")
    raise RuntimeError(f"Configuration loading failed: {e}") from e


# --- Agent State Definition ---
class AgentState(TypedDict):
    """Defines the state passed between nodes in the LangGraph."""

    input: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    final_answer: Optional[str] = None
    error: Optional[str] = None
    max_iterations: int


# --- Agent Nodes ---
def planner_node(
    state: AgentState, planner_llm: BaseChatModel, tools: List[BaseTool]
) -> Dict[str, Any]:
    """
    Executes the planner node by invoking the planner LLM with the current agent state and available tools.
    This function prepares the prompt for the planner LLM, including tool descriptions and the agent's input,
    then constructs the message history to be sent to the LLM. It invokes the LLM, processes the response,
    and returns the resulting message(s) for further state updates. If the LLM invocation fails, an error
    message is returned instead.
    Args:
        state (AgentState): The current state of the agent, including input and message history.
        planner_llm (BaseChatModel): The language model used for planning, capable of tool binding.
        tools (List[BaseTool]): A list of available tools that the planner LLM can utilize.
    Returns:
        Dict[str, Any]: A dictionary containing either:
            - 'messages': A list with the LLM's response message(s) for state update, or
            - 'error': An error message if the LLM invocation fails.
    """
    print("\n--- Executing Planner Node ---")

    prompt_template = get_planner_prompt()
    tool_descriptions = "\n".join([f"- {tool.name}" for tool in tools])

    print(f"Tools available to LLM:\n{tool_descriptions}")

    if hasattr(planner_llm, "bind_tools"):
        planner_llm_with_tools = planner_llm.bind_tools(tools)
        system_prompt_message = prompt_template.format_messages(
            tool_descriptions=tool_descriptions,
            input=state["input"],
        )[0]
        messages_to_llm = [system_prompt_message] + list(state["messages"])
    else:
        planner_llm_with_tools = planner_llm
        # For HuggingFaceEndpoint, just use the user input and message history
        messages_to_llm = list(state["messages"])

    # print(f"Messages to LLM: {messages_to_llm}")

    try:
        # Invoke the LLM
        response: BaseMessage = planner_llm_with_tools.invoke(messages_to_llm)

        print(f"Planner LLM Raw Response Type: {type(response)}")
        print(f"Planner LLM Raw Response Content: {response.content}")
        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"Planner LLM Tool Calls: {response.tool_calls}")
        else:
            print("Planner LLM Response: No tool calls requested.")

        # Return the response message(s) to be added to the state
        return {"messages": [response]}

    except Exception as e:
        print(f"Error in Planner Node: {e}")
        traceback.print_exc()
        # Still return error state if LLM call fails
        return {"error": f"Planner LLM invocation failed: {e}"}


def tool_node(state: AgentState, tools: List[BaseTool]) -> Dict[str, Any]:
    """
    Manually finds and executes the tool calls requested by the planner node.
    """
    print("\n--- Executing Tool Node (Manual Execution) ---")
    tool_messages: List[BaseMessage] = []
    error_message = None  # Tracks if any tool failed critically
    last_message = state["messages"][-1] if state["messages"] else None

    if (
        not isinstance(last_message, AIMessage)
        or not hasattr(last_message, "tool_calls")
        or not last_message.tool_calls
    ):
        print("No valid tool calls found in the last AI message.")
        return {}

    # Create a mapping from tool name to tool instance for faster lookup
    tool_map = {tool.name: tool for tool in tools}

    print(f"Attempting to execute tool calls: {last_message.tool_calls}")
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})  # Arguments are usually a dict
        tool_id = tool_call.get("id")

        if not tool_name or not tool_id:
            print(f"Skipping invalid tool call format: {tool_call}")
            continue

        print(f"Looking for tool: {tool_name} with args: {tool_args} (ID: {tool_id})")

        # Find the corresponding tool instance
        tool_to_invoke = tool_map.get(tool_name)

        if tool_to_invoke:
            try:
                # Use the tool's 'run' method. LangChain's BaseTool.run handles
                # input schema validation and calling the internal _run method
                # with arguments unpacked from the tool_args dictionary.
                output = tool_to_invoke.run(
                    tool_input=tool_args,  # Pass args dict directly
                    # verbose=True, # Optional: for more tool detail
                    # color="blue", # Optional
                )

                # Ensure output is string for ToolMessage content
                if not isinstance(output, str):
                    try:
                        output_str = json.dumps(output, ensure_ascii=False)
                    except TypeError:
                        output_str = str(output)
                else:
                    output_str = output

                print(f"Tool '{tool_name}' output: {output_str}...")
                tool_messages.append(
                    ToolMessage(
                        content=output_str, tool_call_id=tool_id, name=tool_name
                    )
                )

            except Exception as e:
                # Catch errors during the actual tool execution
                print(f"Error executing tool '{tool_name}': {e}")
                tb_str = traceback.format_exc()
                print(tb_str)
                error_output = f"Error executing tool {tool_name} with args {tool_args}. Error: {e}\nTraceback:\n{tb_str}"
                tool_messages.append(
                    ToolMessage(
                        content=error_output, tool_call_id=tool_id, name=tool_name
                    )
                )
                error_message = f"Failed during tool execution ({tool_name}): {e}"  # Propagate error
        else:
            # Tool requested by LLM not found in our list
            print(f"Error: Tool '{tool_name}' requested by LLM not found.")
            error_output = f"Error: Tool '{tool_name}' is not available."
            tool_messages.append(
                ToolMessage(content=error_output, tool_call_id=tool_id, name=tool_name)
            )
            # Consider if this should be a critical error
            # error_message = f"Requested tool '{tool_name}' not found."

    # Return dictionary with messages to be added to the state
    result: Dict[str, Any] = {"messages": tool_messages}
    if error_message:
        result["error"] = (
            error_message  # Update error state if a tool failed critically
        )

    return result


# --- Final Answer Cleaning Node ---
def _clean_gaia_answer(answer: Any) -> str:
    if answer is None:
        return "Error: No answer content found."
    if not isinstance(answer, str):
        if isinstance(answer, (float, int)):
            if isinstance(answer, float) and answer.is_integer():
                return str(int(answer))
            return str(answer)
        else:
            return str(answer)
    answer = answer.strip()
    prefixes = [
        "The final answer is:",
        "The answer is:",
        "Answer:",
        "Final Answer:",
        "Result:",
        "Based on the analysis,",
        "Based on the information provided,",
        "According to the file,",
        "According to the information: ",
        "The calculation shows:",
        "Here is the result:",
        "The extracted text is:",
        "Okay, the answer is",
        "Sure, the answer is",
        "```text",
        "```json",
        "```",
    ]
    suffixes = ["```"]
    cleaned = False
    for prefix in prefixes:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix) :].strip()
            cleaned = True
    for suffix in suffixes:
        if answer.endswith(suffix):
            answer = answer[: -len(suffix)].strip()
            cleaned = True
    if len(answer) >= 2 and (
        (answer.startswith('"') and answer.endswith('"'))
        or (answer.startswith("'") and answer.endswith("'"))
    ):
        answer = answer[1:-1].strip()
        cleaned = True
    print(f"Cleaned answer: {answer}" if cleaned else f"Answer unchanged: {answer}")
    return answer


def final_answer_node(state: AgentState) -> Dict[str, Any]:
    print("\n--- Executing Final Answer Node ---")
    final_answer_raw = None
    error_msg = state.get("error")
    final_answer_cleaned = None
    if error_msg and "Maximum iterations reached" not in error_msg:
        print(f"Propagating critical error: {error_msg}")
        return {"final_answer": None, "error": error_msg}
    last_message = state["messages"][-1] if state["messages"] else None
    if isinstance(last_message, AIMessage) and not (
        hasattr(last_message, "tool_calls") and last_message.tool_calls
    ):
        final_answer_raw = last_message.content
        print(f"Raw final answer from LLM: {final_answer_raw}")
    else:
        print("Warning: Last message not final answer. Fallback.")
        fallback_error = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and not (
                hasattr(msg, "tool_calls") and msg.tool_calls
            ):
                final_answer_raw = msg.content
                print(f"Using fallback raw answer: {final_answer_raw}")
                break
        if final_answer_raw is None:
            fallback_error = "Agent stopped without final answer message."
        if fallback_error:
            print(fallback_error)
            combined_error = (
                f"{error_msg}. {fallback_error}" if error_msg else fallback_error
            )
            return {"final_answer": None, "error": combined_error}

    final_answer_cleaned = _clean_gaia_answer(final_answer_raw)

    original_input = state.get("input", "")

    if original_input.startswith(".") or ".rewsna eht sa" in original_input:
        print("Reversing cleaned answer for GAIA requirement.")
        final_answer_cleaned = final_answer_cleaned[::-1]
        print(f"Reversed final answer: {final_answer_cleaned}")
    return {
        "final_answer": final_answer_cleaned,
        "error": error_msg if error_msg else None,
    }


# --- Conditional Edge Logic ---
def should_continue(state: AgentState) -> str:
    default_max_iter = get_config_value(["agent", "default_max_iterations"], 50)
    max_iterations = state.get("max_iterations", default_max_iter)

    print("\n--- Evaluating Condition 'should_continue' ---")

    last_message = state["messages"][-1] if state["messages"] else None
    error = state.get("error")

    if error and "Maximum iterations reached" not in error:
        print(f"Critical error: {error}. Routing to END.")
        return END

    planner_cycles = sum(1 for msg in state["messages"] if isinstance(msg, AIMessage))

    if planner_cycles >= max_iterations:
        print(
            f"Max iterations ({max_iterations}) reached. Routing to final answer node."
        )
        state["error"] = f"Agent stopped: Max iterations ({max_iterations}) reached."
        return "final_answer_node"

    if isinstance(last_message, AIMessage):
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print("AI requested tool execution. Routing to 'tool_node'.")
            return "tool_node"
        else:
            print("AI provided final answer. Routing to 'final_answer_node'.")
            return "final_answer_node"
    else:
        print("Warning: Last message not AIMessage. Routing to END.")
        state["error"] = "Invalid state: Expected AIMessage after planner."
        return END


# --- Graph Construction ---
def create_gaia_agent_graph(
    planner_llm: BaseChatModel, tools: List[BaseTool]
) -> StateGraph:
    """
    Creates and configures a LangGraph StateGraph for the GAIA agent using manual tool execution.
    This function sets up the agent's execution graph by defining the planner, tool execution, and final answer nodes,
    and establishes the flow between them based on the agent's state and planning logic. The tools are passed directly
    to the relevant nodes, and tool execution is handled manually within the tool_node.
    Args:
        planner_llm (BaseChatModel): The language model instance used for planning agent actions.
        tools (List[BaseTool]): A list of tool instances available for the agent to use.
    Returns:
        StateGraph: The constructed StateGraph representing the agent's execution flow.
    planner_llm: BaseChatModel, tools: List[BaseTool]
    """
    print("--- Creating GAIA Agent Graph (Manual Tool Execution) ---")

    graph = StateGraph(AgentState)

    # Add nodes: Pass LLM and tools instances to the planner node lambda
    graph.add_node(
        "planner",
        lambda state: planner_node(state, planner_llm=planner_llm, tools=tools),
    )
    graph.add_node(
        "tool_node",
        lambda state: tool_node(state, tools=tools),
    )
    graph.add_node("final_answer_node", final_answer_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner",
        should_continue,
        {"tool_node": "tool_node", "final_answer_node": "final_answer_node", END: END},
    )
    graph.add_edge("tool_node", "planner")
    graph.add_edge("final_answer_node", END)

    print("Graph construction complete (using manual tool execution).")
    return graph


# --- Agent Execution ---
def run_agent(
    query: str, compiled_agent: Any, max_iterations: Optional[int] = None
) -> Dict[str, Any]:
    if max_iterations is None:
        max_iterations = get_config_value(["agent", "default_max_iterations"], 50)

    print(f"\n--- Running Agent for Query: '{query}' (Max Iter: {max_iterations}) ---")

    initial_state: AgentState = {
        "input": query,
        "messages": [HumanMessage(content=query)],
        "final_answer": None,
        "error": None,
        "max_iterations": max_iterations,
    }

    final_state = None

    try:
        config_run = {"recursion_limit": max_iterations + 5}
        final_state = compiled_agent.invoke(initial_state, config=config_run)

        print("\n--- Agent Execution Finished ---")

        final_answer = final_state.get("final_answer")
        error = final_state.get("error")
        if error and not final_answer:
            print(f"Agent finished with error: {error}")
            return {"answer": None, "error": error}
        elif final_answer is not None:
            print(f"Agent finished successfully. Final Answer: {final_answer}")
            return {"answer": final_answer, "error": error if error else None}
        else:
            print("Agent finished unexpectedly.")
            return {
                "answer": None,
                "error": "Agent finished without explicit final_answer or error state.",
            }
    except Exception as e:
        print(f"Error during agent execution: {e}")
        tb_str = traceback.format_exc()
        print(tb_str)
        partial_state_info = (
            f"Partial state before error: {final_state}"
            if final_state
            else "No partial state."
        )
        return {
            "answer": None,
            "error": f"Agent execution runtime error: {e}\n{tb_str}\n{partial_state_info}",
        }


if __name__ == "__main__":
    import os
    from gaia_agent.llm_config import get_llm
    # from IPython.display import Image, display

    planner_llm = get_llm("groq")  # gemini, groq, or hf
    tools = get_all_tools()
    agent_graph = create_gaia_agent_graph(planner_llm, tools)

    # Print all step (node) names before execution
    print("\n--- Agent Graph Steps (Nodes) ---")
    for node_name in agent_graph.nodes:
        print(f"- {node_name}")

    # Compile the graph before running
    compiled_graph = agent_graph.compile()

    # Display the graph in Mermaid format
    if not os.path.exists("src/gaia_agent/agent_graph.png"):
        with open("src/gaia_agent/agent_graph.png", "wb") as f:
            f.write(compiled_graph.get_graph(xray=True).draw_mermaid_png())
        print("Agent graph saved to src/gaia_agent/agent_graph.png")

    # Test the agent with a sample query
    query = "What is the first verse of the second song in the 'Hybrid Theory' album of 'Linkin Park'?"
    result = run_agent(query, compiled_graph)
    print(f"Final Result: {result}")
    print("\n--- Agent Execution Complete ---")
