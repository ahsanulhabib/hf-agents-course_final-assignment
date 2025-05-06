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
from pathlib import Path
from typing import Optional

# --- Configuration ---
MAX_ITERATIONS_PER_TASK = 25

# --- Load Environment Variables ---
dotenv.load_dotenv()

# --- Import Agent Components ---
try:
    from src.gaia_agent.llm_config import get_llm
    from src.gaia_agent.tools import get_all_tools
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
        "question": "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.",
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
    {
        "task_id": "task_011",
        "question": "What is the largest planet in our solar system?",
        "expected_answer": "Jupiter",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_012",
        "question": "What is the chemical symbol for gold?",
        "expected_answer": "Au",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_013",
        "question": "What is the boiling point of water in Celsius?",
        "expected_answer": "100",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_014",
        "question": "Translate 'Hello, how are you?' to Spanish.",
        "expected_answer": "Hola, ¿cómo estás?",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_015",
        "question": "What is the Pythagorean theorem?",
        "expected_answer": "a^2 + b^2 = c^2",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_016",
        "question": "What is the capital of Japan?",
        "expected_answer": "Tokyo",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_017",
        "question": "What is the speed of light in vacuum?",
        "expected_answer": "299792458 m/s",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_018",
        "question": "What is the largest mammal in the world?",
        "expected_answer": "Blue whale",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_019",
        "question": "What is the main ingredient in guacamole?",
        "expected_answer": "Avocado",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_020",
        "question": "What is the formula for calculating the area of a circle?",
        "expected_answer": "πr^2",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_021",
        "question": "Who nominated the only Featured Article on English Wikipedia about a dinosaur that was promoted in November 2016?",
        "expected_answer": "FunkMonk",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_022",
        "question": "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?",
        "expected_answer": "3",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_023",
        "question": "Given this table defining * on the set S = {a, b, c, \n        d, e}\n\n|*|a|b|c|d|e|\n|---|---|---|---|---|\n|a|a|b|c|b|d|\n|b|b|c|a|e|c|\n|c|c|a|b|b|a|\n|d|b|e|b|e|d|\n|e|d|b|a|d|c|\n\nprovide the subset of S involved in any possible counter-examples that prove * is not commutative. Provide your answer as a comma separated list of the elements in the set in alphabetical order.",
        "expected_answer": "b,e",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_024",
        "question": "Examine the video at https://www.youtube.com/watch?v=1htKBjuUWec.\n\nWhat does Teal'c say in response to the \n        question \"Isn't that hot?\"",
        "expected_answer": "extremely",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_025",
        "question": "What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials as compiled 08/21/2023?",
        "expected_answer": "Louvrier",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_026",
        "question": "I'm making a grocery list for my mom, but she's a professor of botany and she's a real stickler when it comes to categorizing things. I need to add different foods to different categories on the grocery list, but if I make a mistake, she won't buy anything inserted in the wrong category. Here's the list I have so far:\n\nmilk, eggs, flour, whole bean coffee, Oreos, sweet potatoes, fresh basil, plums, green beans, rice, corn, bell pepper, whole allspice, acorns, broccoli, celery, zucchini, lettuce, peanuts\n\nI need to make headings for the fruits and vegetables. Could you please create \n        a list of just the vegetables from my list? If you could do that, then I can figure out how to categorize the rest of the list into the appropriate categories. But remember that my mom is a real stickler, so make sure that no botanical fruits end up on the vegetable list, or she won't get them when she's at the store. Please alphabetize the list of vegetables, and place each item in a comma separated list.",
        "expected_answer": "broccoli, celery, corn, green beans, lettuce, sweet potatoes, zucchini",
        "has_file": False,
        "file_content": None,
    },
    {
        "task_id": "task_028",
        "question": "What is the name of the actor who played Ray in the Polish-language version of Everybody Loves Raymond?",
        "expected_answer": "Bartłomiej Kasprzykowski",
        "has_file": True,
        "file_content": None,
    },
    {
        "task_id": "task_029",
        "question": "How many at bats did the Yankee with the most walks in the 1977 regular season have that same season?",
        "expected_answer": "589",
        "has_file": True,
        "file_content": None,
    },
    {
        "task_id": "task_030",
        "question": "Where were the Vietnamese specimens described by Kuznetzov in Nedoshivina's 2010 paper eventually deposited? Just give me the city name without abbreviations.",
        "expected_answer": "Saint Petersburg",
        "has_file": True,
        "file_content": None,
    },
    {
        "task_id": "task_031",
        "question": "What country had the least number of athletes \n        at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country \n        code as your answer.",
        "expected_answer": "Cuba",
        "has_file": True,
        "file_content": None,
    },
    {
        "task_id": "task_032",
        "question": "Who are the pitchers with the number before and after Taishō Tamai's number as of July 2023? Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters.",
        "expected_answer": "Yamasaki, Uehara",
        "has_file": True,
        "file_content": None,
    },
    {
        "task_id": "task_033",
        "question": "What is the name of the actor who played Ray in the Polish-language version of Everybody Loves Raymond?",
        "expected_answer": "Claus",
        "has_file": True,
        "file_content": None,
    },
]


def initialize_agent_graph(llm_choice="groq"):
    """Initialize the LangGraph agent components with selected LLM."""
    print(f"\n--- Initializing Agent Graph (LLM: {llm_choice}) ---")
    compiled_agent_graph = None
    try:
        try:
            # Select LLM based on choice and check keys
            planner_llm = get_llm(llm_choice)
        except ValueError as ve:
            print(f"ERROR: {ve}")
            raise ValueError(f"Unsupported LLM choice: {llm_choice}")

        print(f"LLM Initialized: {type(planner_llm)}")

        print("Initializing Tools...")

        try:
            tools = get_all_tools()
            print(f"Tools Initialized: {[tool.name for tool in tools]}")
        except Exception as e:
            print(f"❌ No tools were initialized. Error: {e}")

        try:
            print("Creating Agent Graph...")
            agent_graph = create_gaia_agent_graph(planner_llm, tools)
            print("Compiling Agent Graph...")
            compiled_agent_graph = agent_graph.compile()
            print("--- Agent Graph Initialized and Compiled Successfully ---")
            return compiled_agent_graph
        except Exception as e:
            print(f"ERROR: Failed to create or compile agent graph: {e}")
            traceback.print_exc()
            return None
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
    """Normalize the answer to a more consistent format."""
    if answer is None:
        return ""

    # Unicode normalization (remove accents, etc.)
    def unicode_normalize(text):
        import unicodedata

        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii")
        )

    ans = answer.strip().lower()
    ans = unicode_normalize(ans)

    # Remove enclosing quotes
    ans = ans.strip("\"'`")

    # Remove articles
    # ans = re.sub(r"\b(a|an|the)\b", " ", ans)

    # Remove most punctuation except . , : ; /
    ans = re.sub(r"[^\w\s\.,:;/\-]", "", ans)

    # Collapse multiple spaces
    ans = re.sub(r"\s+", " ", ans)

    # Remove trailing/leading punctuation
    ans = ans.strip(".,:;/ ")

    # Normalize numbers (e.g., "12.0" -> "12")
    try:
        float_ans = float(ans)
        if float_ans.is_integer():
            ans = str(int(float_ans))
        else:
            ans = str(float_ans)
    except ValueError:
        pass

    # Remove common units/phrases (optional, extend as needed)
    units = [
        "m/s",
        "meters per second",
        "km/h",
        "kilometers per hour",
        "°c",
        "celsius",
        "fahrenheit",
        "usd",
        "dollars",
        "euros",
        "kg",
        "kilograms",
        "g",
        "grams",
    ]
    for unit in units:
        ans = ans.replace(unit, "")

    common_phrases = [
        "for example",
        "such as",
        "in other words",
        "that is",
    ]
    for phrase in common_phrases:
        ans = ans.replace(phrase, "")

    # Replace superscripts with ^ notation (e.g., a² -> a^2)
    superscript_map = {
        "²": "^2",
        "³": "^3",
        "⁴": "^4",
        "⁵": "^5",
        "⁶": "^6",
        "⁷": "^7",
        "⁸": "^8",
        "⁹": "^9",
        "¹": "^1",
        "⁰": "^0",
    }
    for sup, rep in superscript_map.items():
        ans = ans.replace(sup, rep)

    # Final cleanup
    ans = ans.strip()
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
    llm_to_test = "gemini"  # Default LLM
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
        results_dir = Path(__file__).parent / "src" / "gaia_test" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_filename = results_dir / f"local_test_results_{llm_to_test}.json"
        try:
            with open(results_filename, "w") as f:
                json.dump(test_results, f, indent=2)
            print(f"\nTest results saved to {results_filename}")
        except Exception as e:
            print(f"\nERROR: Failed to save results: {e}")
    else:
        print("\nAgent initialization failed. Exiting test run.")
    print("\n--- Local Test Finished ---")
