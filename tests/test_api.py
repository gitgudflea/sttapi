from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

TEST_DIR = Path(__file__).parent
AUDIO_PATH = TEST_DIR / "data" / "test_audio2.flac"


def test_api():
    with TestClient(app) as client:
        assert hasattr(app.state, "processor"), "Processor not loaded"
        assert hasattr(app.state, "model"), "Model not loaded"
        print("Model loading test passed")

        with open(AUDIO_PATH, "rb") as f:
            response = client.post(
                "/transcribe",
                files={"file": ("test.flac", f, "audio/flac")}
            )
        assert response.status_code == 200, f"Got {response.status_code}"
        assert "transcription" in response.json()
        print("Valid audio test passed")
        print(response.json())

        response = client.post(
            "/transcribe",
            files={"file": ("bad.txt", b"invalid", "text/plain")}
        )
        assert response.status_code == 415, f"Got {response.status_code}"
        print("Invalid file test passed")

if __name__ == "__main__":
    test_api()
