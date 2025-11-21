import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from voiceBot import app

client = TestClient(app)

CSV_PATH = "flight_questions_variations.csv"

questions = []

# Load ALL variations from the CSV
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    # Loop through all columns
    for col in df.columns:
        questions.extend(df[col].dropna().astype(str).tolist())

    # Remove duplicates
    questions = list(set(questions))

else:
    questions = ["How can I reset my password?"]


@pytest.mark.parametrize("question", questions)
def test_chat_endpoint(question):
    """
    Integration test that sends each question variation
    to the /chat endpoint.
    """
    response = client.post("/chat", data={"message": question})

    # 1. Request OK
    assert response.status_code == 200, f"❌ Failed for: {question}"

    data = response.json()

    # 2. Validate schema
    assert "reply" in data, f"❌ Missing reply for: {question}"
    assert "audio" in data, f"❌ Missing audio for: {question}"

    # 3. Reply must not be empty
    assert isinstance(data["reply"], str)
    assert data["reply"].strip(), f"❌ Empty reply for: {question}"

    # 4. Audio file must exist
    audio_path = data["audio"].lstrip("/")
    assert os.path.exists(audio_path), f"❌ Audio missing for: {question}"

    # 5. Cleanup safely
    if os.path.exists(audio_path):
        os.remove(audio_path)
