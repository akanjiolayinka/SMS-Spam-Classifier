import requests
import argparse

DEF_BASE = "http://127.0.0.1:8000"


def predict(base_url: str, text: str):
    resp = requests.post(f"{base_url}/predict", json={"text": text})
    resp.raise_for_status()
    return resp.json()


def feedback(base_url: str, text: str, true_label: int):
    resp = requests.post(f"{base_url}/feedback", json={"text": text, "true_label": true_label})
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Test spam classifier API")
    parser.add_argument("--base", type=str, default=DEF_BASE, help="Base URL of API")
    parser.add_argument("--text", type=str, required=True, help="Message text to classify")
    parser.add_argument("--feedback-label", type=int, default=None, help="Optional true label to send as feedback (0=ham,1=spam)")
    args = parser.parse_args()

    print("Predicting...")
    pred = predict(args.base, args.text)
    print("Prediction:", pred)

    if args.feedback_label is not None:
        print("Sending feedback...")
        fb = feedback(args.base, args.text, args.feedback_label)
        print("Feedback response:", fb)


if __name__ == "__main__":
    main()
