import torch
import torch.nn as nn
import joblib
from .config import MODEL_PATH, VECTORIZER_PATH


class SpamNet(nn.Module):
	def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
		super().__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(dropout)
		self.fc2 = nn.Linear(hidden_dim, 1)

	def forward(self, x):  # x: [batch, input_dim]
		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.fc2(x)
		return x.squeeze(1)  # [batch]


class TorchSpamClassifier:
	def __init__(self):
		# Load vectorizer
		self.vectorizer = joblib.load(VECTORIZER_PATH)
		input_dim = len(self.vectorizer.get_feature_names_out())

		# Init model & load weights
		self.model = SpamNet(input_dim=input_dim, hidden_dim=128, dropout=0.3)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = device
		state = torch.load(MODEL_PATH, map_location=device)
		self.model.load_state_dict(state)
		self.model.to(device)
		self.model.eval()

	def predict(self, text: str):
		vec = self.vectorizer.transform([text])
		X_dense = vec.toarray().astype("float32")
		X_tensor = torch.from_numpy(X_dense).to(self.device)

		with torch.no_grad():
			logits = self.model(X_tensor)
			prob_spam = torch.sigmoid(logits).item()

		pred = 1 if prob_spam >= 0.5 else 0
		label = "spam" if pred == 1 else "ham"
		return {
			"label": label,
			"spam_probability": prob_spam,
			"raw_pred": pred,
		}


# Instantiate singleton classifier
classifier = TorchSpamClassifier()

