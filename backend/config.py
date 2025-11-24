from pathlib import Path
from dotenv import load_dotenv
import os

# Base project directory (root of repository)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables if .env exists
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
	load_dotenv(ENV_PATH)

MODEL_PATH = BASE_DIR / "models" / "spam_torch_model.pt"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.joblib"
FEEDBACK_PATH = BASE_DIR / "data" / "feedback" / "feedback.csv"

# Optional environment overrides
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODEL_PATH)))
VECTORIZER_PATH = Path(os.getenv("VECTORIZER_PATH", str(VECTORIZER_PATH)))
FEEDBACK_PATH = Path(os.getenv("FEEDBACK_PATH", str(FEEDBACK_PATH)))

