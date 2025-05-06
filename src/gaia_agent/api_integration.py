import requests
from typing import List, Dict, Any


class GAIAApiClient:
    def __init__(self, api_url="https://agents-course-unit4-scoring.hf.space"):
        self.api_url = api_url
        self.questions_url = f"{api_url}/questions"
        self.submit_url = f"{api_url}/submit"
        self.files_url = f"{api_url}/files"

    def get_questions(self) -> List[Dict[str, Any]]:
        """Fetch all evaluation questions"""
        response = requests.get(self.questions_url)
        response.raise_for_status()
        return response.json()

    def get_random_question(self) -> Dict[str, Any]:
        """Fetch a single random question"""
        response = requests.get(f"{self.api_url}/random-question")
        response.raise_for_status()
        return response.json()

    def get_file(self, task_id: str) -> bytes:
        """Download a file for a specific task"""
        response = requests.get(f"{self.files_url}/{task_id}")
        response.raise_for_status()
        return response.content

    def submit_answers(
        self, username: str, agent_code: str, answers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Submit agent answers and get score"""
        data = {"username": username, "agent_code": agent_code, "answers": answers}
        response = requests.post(self.submit_url, json=data)
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    client = GAIAApiClient()
    questions = client.get_questions()
    print("Questions:", questions)

    random_question = client.get_random_question()
    print("Random Question:", random_question)

    # # Example of submitting answers (replace with actual data)
    # username = "test_user"
    # agent_code = "test_agent"
    # answers = [{"question_id": 1, "answer": "Test answer"}]
    # submission_response = client.submit_answers(username, agent_code, answers)
    # print("Submission Response:", submission_response)
