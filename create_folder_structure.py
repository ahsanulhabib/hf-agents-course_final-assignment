import os
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT_NAME = "."

# Define directories relative to the project root
DIRECTORIES_TO_CREATE = [
    "src",
    "src/gaia_agent",
]

# Define empty files to create, relative to the project root
FILES_TO_CREATE = [
    ".gitignore",
    ".env.example",
    "pyproject.toml",
    "requirements.txt",
    "src/gaia_agent/__init__.py",
    "src/gaia_agent/llm_config.py",
    "src/gaia_agent/tools.py",
    "src/gaia_agent/prompts.py",
    "src/gaia_agent/agent_core.py",
    "app.py",
    "local_test.py",
]
# --- End Configuration ---

def generate_structure(root_name: str):
    """Creates the project directory structure and empty files."""

    root_path = Path(root_name)

    print(f"Attempting to create project root: '{root_path}'")
    root_path.mkdir(exist_ok=True) # Create root directory if it doesn't exist
    print(f"'{root_path}' directory ensured.")

    print("\nCreating directories...")
    for rel_dir in DIRECTORIES_TO_CREATE:
        dir_path = root_path / rel_dir # Use pathlib's / operator for joining paths
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  Ensured directory: '{dir_path}'")
        except OSError as e:
            print(f"  ERROR creating directory '{dir_path}': {e}")

    print("\nCreating empty placeholder files...")
    for rel_file in FILES_TO_CREATE:
        file_path = root_path / rel_file
        # Ensure parent directory exists (should be covered by above, but good practice)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch(exist_ok=True) # Creates file if it doesn't exist, does nothing if it does
            print(f"  Ensured file:      '{file_path}'")
        except OSError as e:
            print(f"  ERROR creating file '{file_path}': {e}")

    print(f"\nProject structure generation complete for '{root_name}'.")

if __name__ == "__main__":
    print("--- Project Structure Generator ---")
    generate_structure(PROJECT_ROOT_NAME)
    print("---------------------------------")