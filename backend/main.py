from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import Message, Feedback
from .classifier_service import classifier
from .feedback_service import save_feedback

app = FastAPI(title="SMS Spam Classifier API")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # Adjust for production
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/")
def home():
	return {"message": "Spam classifier is running"}


@app.get("/health")
def health():
	# Basic health info with artifact existence
	from .config import MODEL_PATH, VECTORIZER_PATH, FEEDBACK_PATH
	return {
		"model_loaded": MODEL_PATH.exists(),
		"vectorizer_loaded": VECTORIZER_PATH.exists(),
		"feedback_file": FEEDBACK_PATH.exists(),
		"model_path": str(MODEL_PATH),
		"vectorizer_path": str(VECTORIZER_PATH),
	}


@app.post("/predict")
def predict(message: Message):
	return classifier.predict(message.text)


@app.post("/feedback")
def feedback(info: Feedback):
	save_feedback(info.text, info.true_label)
	return {"status": "feedback saved"}

