import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from voiceBot import app

client = TestClient(app)

# Path to your CSV file
CSV_PATH = "customer_faq.csv"

# Read questions from CSV
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    if "Question" in df.columns:
        questions = df["Question"].dropna().tolist()
    else:
        questions = ["How can I reset my password?"]  # fallback
else:
    questions = ["How can I reset my password?"]  # fallback


@pytest.mark.parametrize("question", questions)
def test_chat_endpoint_with_csv_questions(question):
    """Integration test for /chat using all questions from customer_faq.csv"""
    response = client.post("/chat", data={"message": question})

    # ✅ 1. Ensure request success
    assert response.status_code == 200, f"❌ Failed for: {question}"

    data = response.json()
    assert "reply" in data and "audio" in data, f"❌ Missing keys for: {question}"

    # ✅ 2. Check that reply is non-empty
    assert isinstance(data["reply"], str)
    assert len(data["reply"].strip()) > 0, f"Empty reply for: {question}"

    # ✅ 3. Check that audio file exists
    audio_path = data["audio"].lstrip("/")
    assert os.path.exists(audio_path), f"Audio file missing for: {question}"

    # ✅ 4. Cleanup after test (optional)
    if os.path.exists(audio_path):
        os.remove(audio_path)
