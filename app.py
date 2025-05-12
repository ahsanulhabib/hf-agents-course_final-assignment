import os
from typing import Optional
import re
import gradio as gr
import requests
import pandas as pd
import traceback
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Import GAIA Agent Components ---
try:
    from gaia_agent.llm_config import get_llm
    from gaia_agent.tools import get_all_tools
    from gaia_agent.agent_core import create_gaia_agent_graph, run_agent
    from gaia_agent.config_loader import get_config_value  # Import config loader
    from gaia_agent.logger_config import logger  # Import logger

    IMPORT_SUCCESS = True
    logger.info("Successfully imported GAIA agent components.")
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR_MESSAGE = f"Import Error: {e}. Check src/ structure & requirements.txt.\n{traceback.format_exc()}"
    logger.error(f"ERROR: {IMPORT_ERROR_MESSAGE}")  # Log error instead of print
except Exception as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR_MESSAGE = (
        f"Unexpected error during imports: {e}\n{traceback.format_exc()}"
    )
    logger.error(f"ERROR: {IMPORT_ERROR_MESSAGE}")

# --- Constants ---
DEFAULT_API_URL = (
    "https://agents-course-unit4-scoring.hf.space"  # URL of the scoring server
)

# --- Global Agent Initialization ---
compiled_agent_graph = None
initialization_error = None
agent_initialization_complete = False

if not IMPORT_SUCCESS:
    initialization_error = (
        f"Agent cannot be initialized due to import errors:\n{IMPORT_ERROR_MESSAGE}"
    )
else:
    try:
        logger.info("--- Initializing GAIA Agent Components for Evaluation Runner ---")
        # Get config values
        llm_type = get_config_value(["planner", "default_planner"], "gemini")
        max_iterations_default = get_config_value(
            ["agent", "default_max_iterations"], 25
        )

        # Initialize LLM
        planner_llm = get_llm(llm_type)

        # Initialize Tools
        tools = get_all_tools()
        if not tools:
            raise RuntimeError(
                "Core tools failed to initialize. Check logs and dependencies."
            )

        # Create and Compile Graph
        agent_graph = create_gaia_agent_graph(planner_llm, tools)
        compiled_agent_graph = agent_graph.compile()
        agent_initialization_complete = True
        logger.info("--- GAIA Agent Components Initialized Successfully ---")

    except Exception as e:
        initialization_error = f"Failed to initialize agent components: {e}"
        logger.exception(
            "Error during agent component initialization"
        )  # Log the exception details


def normalize_answer(answer: Optional[str]) -> str:
    """Normalize the answer to a more consistent format."""
    if answer is None:
        return ""

    match = re.search(r"final answer\s*:\s*(.*)", answer, re.IGNORECASE | re.DOTALL)
    if match:
        ans = match.group(1).strip()

    # Unicode normalization (remove accents, etc.)
    def unicode_normalize(text):
        import unicodedata

        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii")
        )

    # ans = ans.strip().lower()
    # ans = unicode_normalize(answer)

    # Remove enclosing quotes
    ans = answer.strip("\"'`")

    # Remove articles
    # ans = re.sub(r"\b(a|an|the)\b", " ", ans)

    # Remove most punctuation except . , : ; /
    # ans = re.sub(r"[^\w\s\.,:;/\-]", "", ans)

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
    # for unit in units:
    #     ans = ans.replace(unit, "")

    common_phrases = [
        "for example",
        "such as",
        "in other words",
        "that is",
        "based on the search results,",
        "according to the search results,",
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


# --- Main Evaluation Function ---
def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the GAIA agent on them, submits all answers,
    and displays the results.
    """
    logger.info("Starting evaluation run...")

    # 0. Check Agent Initialization and Login Status
    if initialization_error:
        logger.error(
            f"Cannot run evaluation due to initialization error: {initialization_error}"
        )
        return f"Agent Initialization Error:\n{initialization_error}", None
    if not agent_initialization_complete or not compiled_agent_graph:
        logger.error("Cannot run evaluation: Agent graph not compiled.")
        return (
            "Agent graph was not compiled successfully during startup. Check logs.",
            None,
        )
    if not profile:
        logger.warning("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    username = profile.username
    logger.info(f"User logged in: {username}")

    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID")  # Get the SPACE_ID for sending link to the code
    if space_id:
        agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
        logger.info(f"Agent code link: {agent_code}")
    else:
        agent_code = "local_run_or_space_id_not_set"
        logger.warning("SPACE_ID not found. Agent code link will be generic.")

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Fetch Questions
    logger.info(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=30)  # Increased timeout
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data or not isinstance(questions_data, list):
            logger.error(
                f"Fetched questions list is empty or invalid format: {questions_data}"
            )
            return "Fetched questions list is empty or invalid format.", None
        logger.info(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        logger.exception("Error fetching questions")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response from questions endpoint: {e}")
        logger.error(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        logger.exception("An unexpected error occurred fetching questions")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 2. Run GAIA Agent
    results_log = []
    answers_payload = []
    logger.info(f"Running GAIA agent on {len(questions_data)} questions...")
    max_iter = get_config_value(
        ["agent", "default_max_iterations"], 25
    )  # Use config default

    for i, item in enumerate(questions_data):
        task_id = item.get("task_id")
        question_text = item.get("question")
        logger.info(
            f"--- Processing Task {i + 1}/{len(questions_data)} (ID: {task_id}) ---"
        )
        logger.debug(f"Question: {question_text}...")  # Log truncated question

        if not task_id or question_text is None:
            logger.warning(f"Skipping item with missing task_id or question: {item}")
            continue

        submitted_answer = (
            "AGENT_ERROR: Unknown failure"  # Default in case of unexpected exit
        )
        try:
            # Use the imported run_agent function with the compiled graph
            agent_result = run_agent(
                query=question_text,
                compiled_agent=compiled_agent_graph,
                max_iterations=max_iter,
            )
            # Extract answer, handle potential errors from the agent run itself
            if agent_result.get("error"):
                logger.warning(
                    f"Agent returned an error for task {task_id}: {agent_result['error']}"
                )
                submitted_answer = (
                    f"AGENT_NOTE: {agent_result['error']}"  # Submit error note
                )
                # If agent provided an answer despite error, use it, otherwise submit the note
                if agent_result.get("answer") is not None:
                    submitted_answer = agent_result["answer"]
                    logger.info(
                        f"Agent provided answer despite error: {submitted_answer[:100]}..."
                    )
                else:
                    logger.warning("Submitting agent error note as answer.")

            elif agent_result.get("answer") is not None:
                submitted_answer = normalize_answer(agent_result["answer"])
                logger.info(f"Agent answer generated: {submitted_answer[:100]}...")
            else:
                logger.error(
                    f"Agent returned neither answer nor error for task {task_id}."
                )
                submitted_answer = "AGENT_ERROR: No answer or error returned."

            answers_payload.append(
                {"task_id": task_id, "submitted_answer": submitted_answer}
            )
            results_log.append(
                {
                    "Task ID": task_id,
                    "Question": question_text,
                    "Submitted Answer": submitted_answer,
                }
            )

        except Exception as e:
            logger.exception(f"Critical error running agent on task {task_id}")
            submitted_answer = f"AGENT_CRITICAL_ERROR: {e}"
            # Still try to log and potentially submit the error
            answers_payload.append(
                {"task_id": task_id, "submitted_answer": submitted_answer}
            )
            results_log.append(
                {
                    "Task ID": task_id,
                    "Question": question_text,
                    "Submitted Answer": submitted_answer,
                }
            )

    if not answers_payload:
        logger.error("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 3. Prepare Submission
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    logger.info(status_update)

    # 4. Submit
    logger.info(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(
            submit_url, json=submission_data, timeout=90
        )  # Increased timeout
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        logger.info("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        logger.error(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        logger.error(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        logger.exception("Submission network error")
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        logger.exception("Unexpected submission error")
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
logger.info("Building Gradio Interface...")
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**
        1.  This app uses the GAIA agent defined in the `src/gaia_agent` directory.
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run the agent, submit answers, and see the score.
        ---
        **Notes:**
        *   Clicking the submit button can take **several minutes** as the agent processes all questions sequentially.
        *   Ensure the necessary API keys (e.g., `GOOGLE_API_KEY`, potentially others for tools) are set as Secrets in your HF Space.
        *   Check the application logs (link usually available in the HF Space menu) for detailed agent execution traces.
        """
    )

    # Display initialization status
    if initialization_error:
        gr.Markdown(
            f"⚠️ **Agent Initialization Failed:**\n```\n{initialization_error}\n```"
        )
    elif not IMPORT_SUCCESS:
        gr.Markdown(f"⚠️ **Agent Import Failed:**\n```\n{IMPORT_ERROR_MESSAGE}\n```")
    else:
        gr.Markdown("✅ Agent components initialized successfully.")

    gr.LoginButton()

    run_button = gr.Button(
        "Run Evaluation & Submit All Answers",
        variant="primary",
        interactive=agent_initialization_complete,
    )  # Disable if init failed

    status_output = gr.Textbox(
        label="Run Status / Submission Result", lines=5, interactive=False
    )
    results_table = gr.DataFrame(label="Questions and Submitted Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table],
        # Add api_name="run_evaluation" if you want to call this via API
    )

if __name__ == "__main__":
    logger.info("\n" + "-" * 30 + " App Starting " + "-" * 30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        logger.info(f"SPACE_HOST found: {space_host_startup}")
    else:
        logger.info("SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:
        logger.info(
            f"SPACE_ID found: {space_id_startup} | Repo: https://huggingface.co/spaces/{space_id_startup}"
        )
    else:
        logger.info("SPACE_ID environment variable not found (running locally?).")
    logger.info("-" * (60 + len(" App Starting ")) + "\n")

    logger.info("Launching Gradio Interface for GAIA Agent Evaluation...")
    # Set server_name="0.0.0.0" to allow access from network if needed
    demo.launch(debug=False, share=False, server_name="127.0.0.1")
