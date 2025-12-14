# SMS Spam Classifier (PyTorch + FastAPI)

Production-style SMS/email spam classifier:

- PyTorch neural network trained on the classic SMS Spam Collection dataset
- TF-IDF features (scikit-learn)
- FastAPI backend with `/predict` and `/feedback` endpoints
- Retraining pipeline that merges user feedback into the dataset

## Features
- Data loading, cleaning, train/test split
- TF-IDF vectorization (1–2 grams, max 5000 features)
- Lightweight fully-connected PyTorch binary classifier
- Metrics & confusion matrix
- FastAPI inference + feedback logging
- Retrain script (`retrain.py`) to incorporate new labeled data

## Tech Stack
| Layer | Tools |
|-------|-------|
| Language | Python 3.10+ |
| ML / NLP | PyTorch, scikit-learn, pandas, numpy |
| Serving | FastAPI, Uvicorn |
| Utilities | python-dotenv, tqdm, joblib |

## Project Structure
```
SMS_SPAM_CLASSIFIER/
├── backend/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── classifier_service.py
│   ├── feedback_service.py
│   └── main.py
├── data/
│   ├── raw/spam.csv
│   ├── processed/train.csv
│   ├── processed/test.csv
│   └── feedback/feedback.csv
├── models/
│   ├── spam_torch_model.pt
│   └── tfidf_vectorizer.joblib
├── notebooks/01_train_model.ipynb
├── retrain.py
├── requirements.txt
└── README.md
```

## Setup
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Training (Notebook)
Open `notebooks/01_train_model.ipynb` and run cells sequentially. Artifacts are saved to `models/`.

## Run API Locally
```powershell
uvicorn backend.main:app --reload
```
Endpoints:
- `GET /` – health/message
- `POST /predict` – body: `{ "text": "Some message" }`
- `POST /feedback` – body: `{ "text": "Message", "true_label": 0 }`

### Example (PowerShell)
```powershell
$body = @{ text = "Congratulations you won cash!" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -Body $body -ContentType application/json
```

## Retrain with Feedback
Each feedback submission appends a line to `data/feedback/feedback.csv` formatted as:
```
<label>\t<text>
```
Run:
```powershell
python retrain.py --epochs 8 --lr 0.001
```
Outputs overwrite `models/spam_torch_model.pt` and `models/tfidf_vectorizer.joblib` (override paths via env vars or CLI args).

## Environment Overrides (.env)
Optional `.env` keys:
```
MODEL_PATH=custom_models/spam_torch_model.pt
VECTORIZER_PATH=custom_models/tfidf_vectorizer.joblib
FEEDBACK_PATH=data/feedback/feedback.csv
```

## Docker
Build & run (after adding Dockerfile):
```powershell
docker build -t spam-api .
docker run -p 8000:8000 spam-api
```

## Deployment (Render Example)
Build command:
```
pip install -r requirements.txt
```
Start command:
```
uvicorn backend.main:app --host 0.0.0.0 --port 10000
```

## Future Improvements
- Add caching layer for vectorizer/model
- Batched prediction endpoint
- Authentication / rate limiting
- Model versioning with timestamps
- Basic frontend / streamlit dashboard

### License
Proprietary / internal (adjust as needed).


