from joblib import load
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, hamming_loss

from train import load_data


def main(repo_path):
    test_csv_path = repo_path / "data/prepared/test.csv"
    test_data, labels = load_data(test_csv_path)
    model = load(repo_path / "model/model.joblib")
    predictions = model.predict(test_data)
    print(labels[:10], predictions[:10])
    accuracy = accuracy_score(labels, predictions)
    hamming = hamming_loss(labels, predictions)
    metrics = {"accuracy": accuracy,
               "hamming_loss": hamming}
    accuracy_path = repo_path / "metrics/accuracy.json"
    accuracy_path.write_text(json.dumps(metrics))


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
