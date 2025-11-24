import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

from backend.classifier_service import SpamNet  # reuse architecture
from backend.config import BASE_DIR, MODEL_PATH, VECTORIZER_PATH, FEEDBACK_PATH


def load_raw_dataset(raw_path: Path) -> pd.DataFrame:
	df = pd.read_csv(raw_path, encoding="latin-1")
	if "v1" in df.columns and "v2" in df.columns:
		df = df[["v1", "v2"]]
		df.columns = ["label", "text"]
	df = df.dropna(subset=["label", "text"])
	df["label"] = df["label"].map({"ham": 0, "spam": 1})
	return df


def load_feedback(feedback_path: Path) -> pd.DataFrame:
	if not feedback_path.exists():
		return pd.DataFrame(columns=["label", "text"])  # empty
	rows = []
	with open(feedback_path, encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			parts = line.split("\t", 1)
			if len(parts) != 2:
				continue
			label_str, text = parts
			try:
				label = int(label_str)
			except ValueError:
				continue
			rows.append((label, text))
	return pd.DataFrame(rows, columns=["label", "text"])


def train_model(df: pd.DataFrame, epochs: int, lr: float, hidden_dim: int, dropout: float, batch_size: int):
	X_train, X_val, y_train, y_val = train_test_split(
		df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
	)

	vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
	X_train_vec = vectorizer.fit_transform(X_train)
	X_val_vec = vectorizer.transform(X_val)

	X_train_dense = X_train_vec.toarray().astype("float32")
	X_val_dense = X_val_vec.toarray().astype("float32")
	y_train_arr = y_train.values.astype("float32")
	y_val_arr = y_val.values.astype("float32")

	input_dim = X_train_dense.shape[1]

	class SimpleDataset(torch.utils.data.Dataset):
		def __init__(self, X, y):
			self.X = X
			self.y = y
		def __len__(self):
			return self.X.shape[0]
		def __getitem__(self, idx):
			return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)

	train_ds = SimpleDataset(X_train_dense, y_train_arr)
	val_ds = SimpleDataset(X_val_dense, y_val_arr)
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = SpamNet(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
	criterion = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	def eval_epoch():
		model.eval()
		correct = 0
		total = 0
		with torch.no_grad():
			for Xb, yb in val_loader:
				Xb = Xb.to(device)
				yb = yb.to(device)
				logits = model(Xb)
				probs = torch.sigmoid(logits)
				preds = (probs >= 0.5).float()
				correct += (preds == yb).sum().item()
				total += yb.size(0)
		return correct / total if total else 0.0

	for epoch in range(1, epochs + 1):
		model.train()
		total_loss = 0.0
		for Xb, yb in train_loader:
			Xb = Xb.to(device)
			yb = yb.to(device)
			optimizer.zero_grad()
			logits = model(Xb)
			loss = criterion(logits, yb)
			loss.backward()
			optimizer.step()
			total_loss += loss.item() * Xb.size(0)
		avg_loss = total_loss / len(train_loader.dataset)
		val_acc = eval_epoch()
		print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

	return model, vectorizer


def main():
	parser = argparse.ArgumentParser(description="Retrain spam classifier incorporating feedback")
	parser.add_argument("--raw", type=str, default=str(BASE_DIR / "data" / "raw" / "spam.csv"), help="Path to original raw dataset")
	parser.add_argument("--epochs", type=int, default=8)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--hidden-dim", type=int, default=128)
	parser.add_argument("--dropout", type=float, default=0.3)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--out-model", type=str, default=str(MODEL_PATH))
	parser.add_argument("--out-vectorizer", type=str, default=str(VECTORIZER_PATH))
	parser.add_argument("--timestamp", action="store_true", help="Store timestamped copies alongside main artifacts")
	args = parser.parse_args()

	raw_path = Path(args.raw)
	print(f"Loading raw dataset from {raw_path}")
	raw_df = load_raw_dataset(raw_path)
	fb_df = load_feedback(FEEDBACK_PATH)
	if len(fb_df):
		print(f"Loaded {len(fb_df)} feedback rows")
	else:
		print("No feedback found; training on original data only")

	combined_df = pd.concat([raw_df, fb_df], ignore_index=True)
	print(f"Combined dataset size: {len(combined_df)}")

	model, vectorizer = train_model(
		combined_df,
		epochs=args.epochs,
		lr=args.lr,
		hidden_dim=args.hidden_dim,
		dropout=args.dropout,
		batch_size=args.batch_size,
	)

	torch.save(model.state_dict(), args.out_model)
	joblib.dump(vectorizer, args.out_vectorizer)
	print(f"Saved model to {args.out_model}")
	print(f"Saved vectorizer to {args.out_vectorizer}")

	if args.timestamp:
		import datetime
		ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
		model_ts = MODEL_PATH.parent / f"spam_torch_model_{ts}.pt"
		vect_ts = VECTORIZER_PATH.parent / f"tfidf_vectorizer_{ts}.joblib"
		torch.save(model.state_dict(), model_ts)
		joblib.dump(vectorizer, vect_ts)
		print(f"Timestamped copies: {model_ts.name}, {vect_ts.name}")


if __name__ == "__main__":
	main()

