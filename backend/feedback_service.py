from .config import FEEDBACK_PATH

# Ensure directory exists
FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

def save_feedback(text: str, true_label: int) -> None:
	# Sanitize newlines
	safe_text = text.replace("\n", " ")
	# Append tab-separated feedback (label<TAB>text)
	with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
		f.write(f"{true_label}\t{safe_text}\n")

