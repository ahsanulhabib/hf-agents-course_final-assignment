"""
Local test script for the GAIA LangGraph agent using real API keys.
"""

import os
import re
import io
import sys
import json
import tempfile
import traceback
import dotenv
import time
import contextlib
from typing import Optional

# --- Configuration ---
MAX_ITERATIONS_PER_TASK = 15

# --- Load Environment Variables ---
dotenv.load_dotenv()
print("Loaded environment variables from .env")

# --- Import Agent Components ---
try:
    from src.gaia_agent.llm_config import (
        get_gemini_llm,
        get_groq_llm,
        get_hf_inference_llm,
    )
    from src.gaia_agent.tools import get_tools
    from src.gaia_agent.agent_core import create_gaia_agent_graph, run_agent

    print("Successfully imported agent components.")
except ImportError as e:
    print(f"ERROR: Import failed: {e}. Check src/ structure & requirements.txt.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Unexpected import error: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Sample Questions ---
SAMPLE_QUESTIONS = [
    {
        "task_id": "task_001",
        "question": "What is the capital of France?",
        "expected_answer": "Paris",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_002",
        "question": "What is the square root of 144?",
        "expected_answer": "12",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_003",
        "question": "If a train travels at 60 miles per hour, how far will it travel in 2.5 hours?",
        "expected_answer": "150",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_004",
        "question": ".rewsna eht sa 'thgir' drow eht etirw ,tfel fo etisoppo eht si tahW",
        "expected_answer": "right",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_005",
        "question": "Analyze the data in the attached CSV file and tell me the total sales for the month of January.",
        "expected_answer": "10250.75",
        "has_file": True,
        "file_content": "Date,Product,Quantity,Price,Total\n2023-01-05,Widget A,10,25.99,259.90\n2023-01-12,Widget B,5,45.50,227.50\n2023-01-15,Widget C,20,50.25,1005.00\n2023-01-20,Widget A,15,25.99,389.85\n2023-01-25,Widget B,8,45.50,364.00\n2023-01-28,Widget D,100,80.04,8004.50",
    },
    {
        "task_id": "task_007",
        "question": "How many studio albums were published by Mercedes Sosa between 1972 and 1985?",
        "expected_answer": "12",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_009_yt",
        "question": "Get the title and uploader for the youtube video https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "expected_answer": "Rick Astley - Never Gonna Give You Up (Official Music Video)",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_010_wiki",
        "question": "What is LangGraph according to Wikipedia?",
        "expected_answer": "LangGraph is a library",
        "has_file": False,
        "file_content": None,
    },
]


def initialize_agent_graph(llm_choice="gemini"):
    """Initialize the LangGraph agent components with selected LLM."""
    print(f"\n--- Initializing Agent Graph (LLM: {llm_choice}) ---")
    compiled_agent_graph = None
    try:
        # Select LLM based on choice and check keys
        planner_llm = None
        if llm_choice == "gemini":
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found.")
            print(f"Using Google API Key: ...{google_api_key[-4:]}")
            planner_llm = get_gemini_llm(api_key=google_api_key)
        elif llm_choice == "groq":
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found.")
            print(f"Using Groq API Key: ...{groq_api_key[-4:]}")
            planner_llm = get_groq_llm(api_key=groq_api_key)
        elif llm_choice == "hf":
            hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if not hf_api_key:
                raise ValueError("HUGGINGFACEHUB_API_TOKEN not found.")
            print(f"Using HF API Token: ...{hf_api_key[-4:]}")
            # You might want to specify a better model here if needed
            planner_llm = get_hf_inference_llm(
                api_key=hf_api_key, repo_id="mistralai/Mistral-7B-Instruct-v0.2"
            )
        else:
            raise ValueError(f"Unsupported LLM choice: {llm_choice}")

        print(f"LLM Initialized: {type(planner_llm)}")

        print("Initializing Tools...")
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        tools = get_tools(tavily_api_key=tavily_api_key)
        if not tools:
            raise RuntimeError("Failed to initialize tools.")
        print(f"Tools Initialized: {[tool.name for tool in tools]}")

        print("Creating Agent Graph...")
        agent_graph = create_gaia_agent_graph(planner_llm, tools)
        print("Compiling Agent Graph...")
        compiled_agent_graph = agent_graph.compile()
        print("--- Agent Graph Initialized and Compiled Successfully ---")
        return compiled_agent_graph
    except Exception as e:
        print(f"ERROR: Failed during agent initialization: {e}")
        traceback.print_exc()
        return None


def save_temp_file(task_id: str, content: str, extension: str = ".txt") -> str:
    """Save content to a temporary file with a specific extension."""
    temp_dir = tempfile.gettempdir()
    filename = f"gaia_test_{task_id}{extension}"
    file_path = os.path.join(temp_dir, filename)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created temporary file for task {task_id}: {file_path}")
        return file_path
    except Exception as e:
        print(f"ERROR: Failed to create temp file {file_path}: {e}")
        return ""


def normalize_answer(answer: Optional[str]) -> str:
    """Basic normalization for comparison."""
    if answer is None:
        return ""
    ans = answer.lower().strip()
    ans = re.sub(r"[,\$\%\.\s]|miles|albums", "", ans)  # Remove common noise
    try:  # Normalize numbers
        float_ans = float(ans)
        if float_ans.is_integer():
            ans = str(int(float_ans))
    except ValueError:
        pass
    return ans


def flexible_check(agent_answer: Optional[str], expected_answer: str) -> bool:
    """More flexible check for answers where exact match is hard."""
    if agent_answer is None:
        return False
    # Check if expected keywords are present in agent answer (case-insensitive)
    expected_keywords = re.findall(r"\b\w+\b", expected_answer.lower())
    agent_lower = agent_answer.lower()
    return all(keyword in agent_lower for keyword in expected_keywords)


def run_tests(compiled_agent_graph):
    """Run tests using the compiled LangGraph agent."""
    if not compiled_agent_graph:
        print("Agent graph not initialized.")
        return []
    results = []
    correct_count = 0
    total_count = len(SAMPLE_QUESTIONS)
    temp_files_created = []

    for idx, task_data in enumerate(SAMPLE_QUESTIONS):
        task_id = task_data["task_id"]
        question = task_data["question"]
        expected = task_data["expected_answer"]
        is_flexible = "wiki" in task_id or "yt" in task_id
        print(f"\n{'=' * 80}\nRunning Task {idx + 1}/{total_count} (ID: {task_id})")
        print(
            f"Question: {question}\nExpected Answer: {expected}{' (flexible check)' if is_flexible else ''}"
        )

        current_question = question
        if task_data["has_file"] and task_data["file_content"]:
            ext = ".csv" if "csv" in question.lower() else ".txt"
            local_temp_path = save_temp_file(
                task_id, task_data["file_content"], extension=ext
            )
            if local_temp_path:
                temp_files_created.append(local_temp_path)
                current_question = (
                    f"{question}\n\nFile saved locally at: {local_temp_path}"
                )
                print("Modified question with file path.")
            else:
                print("WARNING: Failed to create temp file.")

        start_time = time.time()
        agent_result = None
        agent_answer = None
        agent_error = None
        run_log_capture = io.StringIO()  # Capture prints during agent run
        with (
            contextlib.redirect_stdout(run_log_capture),
            contextlib.redirect_stderr(run_log_capture),
        ):
            try:
                agent_result = run_agent(
                    query=current_question,
                    compiled_agent=compiled_agent_graph,
                    max_iterations=MAX_ITERATIONS_PER_TASK,
                )
                agent_answer = agent_result.get("answer")
                agent_error = agent_result.get("error")
            except Exception as e:
                print(f"ERROR: Agent execution exception: {e}")
                agent_error = f"Runtime Exception: {e}\n{traceback.format_exc()}"
        run_logs = run_log_capture.getvalue()
        end_time = time.time()
        duration = end_time - start_time

        print(f"Agent Raw Answer: {agent_answer}")
        if agent_error:
            print(f"Agent Error/Note: {agent_error}")
        print(f"Execution Time: {duration:.2f} seconds")

        is_correct = False
        if agent_answer is not None:
            if is_flexible:
                is_correct = flexible_check(agent_answer, expected)
            else:
                is_correct = normalize_answer(agent_answer) == normalize_answer(
                    expected
                )

        if is_correct:
            correct_count += 1
            print("✅ CORRECT")
        else:
            print("❌ INCORRECT")
            print(f"  -> Normalized Agent: '{normalize_answer(agent_answer)}'")
            print(f"  -> Normalized Expected: '{normalize_answer(expected)}'")

        results.append(
            {
                "task_id": task_id,
                "question": question,
                "question_sent": current_question,
                "expected": expected,
                "raw_answer": agent_answer,
                "normalized_answer": normalize_answer(agent_answer),
                "error": agent_error,
                "is_correct": is_correct,
                "duration_seconds": duration,
                "run_logs": run_logs,  # Include captured logs
            }
        )

    print("\nCleaning up temporary files...")
    for f_path in temp_files_created:
        try:
            os.remove(f_path)
            print(f"Removed: {f_path}")
        except OSError as e:
            print(f"Warning: Could not remove temp file {f_path}: {e}")

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(
        f"\n{'=' * 80}\nTest Run Summary:\nTotal Questions: {total_count}\nCorrect Answers: {correct_count}\nAccuracy: {accuracy:.1f}%\n{'=' * 80}"
    )
    return results


if __name__ == "__main__":
    print("--- Starting Local GAIA Agent Test ---")
    # Allow choosing LLM via command line argument, default to gemini
    llm_to_test = "gemini"
    if len(sys.argv) > 1 and sys.argv[1] in ["gemini", "groq", "hf"]:
        llm_to_test = sys.argv[1]
        print(f"Testing with LLM specified via argument: {llm_to_test}")
    else:
        print(
            f"Testing with default LLM: {llm_to_test} (Use 'python local_test.py [gemini|groq|hf]' to choose)"
        )

    compiled_agent = initialize_agent_graph(llm_choice=llm_to_test)
    if compiled_agent:
        test_results = run_tests(compiled_agent)
        results_filename = f"local_test_results_{llm_to_test}.json"
        try:
            with open(results_filename, "w") as f:
                json.dump(test_results, f, indent=2)
            print(f"\nTest results saved to {results_filename}")
        except Exception as e:
            print(f"\nERROR: Failed to save results: {e}")
    else:
        print("\nAgent initialization failed. Exiting test run.")
    print("\n--- Local Test Finished ---")
