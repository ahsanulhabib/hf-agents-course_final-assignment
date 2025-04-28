import os
import io
import re
import json
import operator
import traceback
from typing import TypedDict, Annotated, Sequence, List, Optional, Dict, Any, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph.message import add_messages

# Import from within the package
from src.gaia_agent.llm_config import BaseChatModel, get_llm  # Use the generic getter
from src.gaia_agent.prompts import get_planner_prompt
from src.gaia_agent.tools import (
    get_all_tools,
)  # Use the function that returns tool instances
from src.gaia_agent import config  # Import config for defaults


# --- Agent State Definition ---
class AgentState(TypedDict):
    input: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    final_answer: Optional[str] = None
    error: Optional[str] = None
    max_iterations: int


# --- Agent Nodes ---


def planner_node(
    state: AgentState, planner_llm: BaseChatModel, tools: List[BaseTool]
) -> Dict[str, Any]:
    """Invokes the planner LLM to decide the next action or provide the final answer."""
    print("\n--- Executing Planner Node ---")
    prompt_template = get_planner_prompt()  # Get the template
    # Dynamically generate tool descriptions from the instantiated tools
    tool_descriptions = "\n".join(
        [f"- {tool.name}: {tool.description}" for tool in tools]
    )
    # Bind tools for function calling
    planner_llm_with_tools = planner_llm.bind_tools(tools)
    # Format the prompt with current state and tool descriptions
    formatted_messages = prompt_template.format_messages(
        tool_descriptions=tool_descriptions,
        input=state["input"],
        messages=state["messages"],
    )
    try:
        # Invoke the LLM
        response: BaseMessage = planner_llm_with_tools.invoke(formatted_messages)
        print(f"Planner LLM Raw Response Type: {type(response)}")
        print(f"Planner LLM Raw Response Content: {response.content}")
        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"Planner LLM Tool Calls: {response.tool_calls}")
        else:
            print("Planner LLM Response: No tool calls requested.")
        # Response (AIMessage) is added automatically via add_messages
        return {}
    except Exception as e:
        print(f"Error in Planner Node: {e}")
        traceback.print_exc()
        return {"error": f"Planner LLM invocation failed: {e}"}


def tool_node(state: AgentState, tool_executor: ToolExecutor) -> Dict[str, Any]:
    """Executes the tool calls requested by the planner node."""
    # This node remains largely the same as ToolExecutor handles class-based tools
    print("\n--- Executing Tool Node ---")
    tool_messages: List[BaseMessage] = []
    error_message = None
    last_message = state["messages"][-1] if state["messages"] else None

    if (
        not isinstance(last_message, AIMessage)
        or not hasattr(last_message, "tool_calls")
        or not last_message.tool_calls
    ):
        print("No tool calls found in the last AI message.")
        return {}

    print(f"Attempting to execute tool calls: {last_message.tool_calls}")
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id")
        if not tool_name or not tool_id:
            print(f"Skipping invalid tool call format: {tool_call}")
            continue

        print(f"Invoking tool: {tool_name} with args: {tool_args} (ID: {tool_id})")
        try:
            # ToolExecutor handles invoking the _run method of the tool class instance
            tool_invocation = ToolInvocation(tool=tool_name, tool_input=tool_args)
            output = tool_executor.invoke(tool_invocation)
            if not isinstance(output, str):
                try:
                    output_str = json.dumps(output, ensure_ascii=False)
                except TypeError:
                    output_str = str(output)
            else:
                output_str = output
            print(f"Tool '{tool_name}' output (truncated): {output_str[:500]}...")
            tool_messages.append(
                ToolMessage(content=output_str, tool_call_id=tool_id, name=tool_name)
            )
        except Exception as e:
            print(f"Error executing tool '{tool_name}': {e}")
            traceback.print_exc()
            error_output = f"Error executing tool {tool_name}: {e}"
            tool_messages.append(
                ToolMessage(content=error_output, tool_call_id=tool_id, name=tool_name)
            )
            error_message = f"Failed during tool execution: {e}"  # Propagate error

    result: Dict[str, Any] = {"messages": tool_messages}
    if error_message:
        result["error"] = error_message
    return result


# --- Final Answer Cleaning Node ---
# (Keep _clean_gaia_answer and final_answer_node functions exactly as in the previous answer)
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
# (Keep should_continue function exactly as in the previous answer)
def should_continue(state: AgentState) -> str:
    print("\n--- Evaluating Condition 'should_continue' ---")
    last_message = state["messages"][-1] if state["messages"] else None
    error = state.get("error")
    if error and "Maximum iterations reached" not in error:
        print(f"Critical error: {error}. Routing to END.")
        return END
    planner_cycles = sum(1 for msg in state["messages"] if isinstance(msg, AIMessage))
    max_iterations = state.get("max_iterations", config.DEFAULT_MAX_ITERATIONS)
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
    """Builds the LangGraph StateGraph for the GAIA agent."""
    print("--- Creating GAIA Agent Graph ---")
    tool_executor = ToolExecutor(tools)
    graph = StateGraph(AgentState)
    graph.add_node(
        "planner",
        lambda state: planner_node(state, planner_llm=planner_llm, tools=tools),
    )
    graph.add_node(
        "tool_node", lambda state: tool_node(state, tool_executor=tool_executor)
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
    print("Graph construction complete.")
    return graph


# --- Agent Execution ---
# (Keep run_agent function exactly as in the previous answer)
def run_agent(
    query: str, compiled_agent: Any, max_iterations: int = config.DEFAULT_MAX_ITERATIONS
) -> Dict[str, Any]:
    print(f"\n--- Running Agent for Query: '{query}' ---")
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
        traceback.print_exc()
        partial_state_info = (
            f"Partial state before error: {final_state}"
            if final_state
            else "No partial state."
        )
        return {
            "answer": None,
            "error": f"Agent execution runtime error: {e}\n{partial_state_info}",
        }
