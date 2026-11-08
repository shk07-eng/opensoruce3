"""Compute cat/dog metrics for Oxford-IIIT Pet evaluation outputs.

The source dataset has breed labels, while the ImageNet model predicts
ImageNet labels.  Both are normalized to the two coarse species classes.
"""
import argparse, json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

CAT_BREEDS = {
    "abyssinian", "bengal", "birman", "bombay", "british_shorthair",
    "egyptian_mau", "havana", "japanese_bobtail", "maine_coon", "persian",
    "ragdoll", "russian_blue", "siamese", "sphynx",
}

CAT_WORDS = {"tabby", "tiger_cat", "persian_cat", "siamese_cat", "egyptian_cat"}

def species_from_breed(value):
    name = value.lower().strip().replace(" ", "_")
    return "cat" if name in CAT_BREEDS else "dog"

def species_from_prediction(value):
    name = value.lower().strip().replace(" ", "_")

    if name == "cat":
        return "cat"
    if name == "dog":
        return "dog"

    return "cat" if name in CAT_WORDS or name.endswith("_cat") else "dog" if name in {
        "boxer", "chihuahua", "collie", "dingo", "german_shepherd",
        "great_dane", "malinois", "pug", "rottweiler", "samoyed",
        "schnauzer", "terrier", "toy_poodle", "standard_poodle",
        "wolfhound", "labrador_retriever",
    } or any(word in name for word in (
        "retriever", "spaniel", "terrier", "hound", "poodle", "shepherd"
    )) else "other"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, default=Path("mlops-results"))
    args = p.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload["predictions"]
    y_true = [species_from_breed(r["label"]) for r in rows]
    y_pred = [species_from_prediction(r["predicted"]) for r in rows]
    metric_labels = ["cat", "dog", "other"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=metric_labels, average="macro", zero_division=0
    )
    result = {
        "model_version": payload.get("model_version"),
        "count": len(rows),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "metric_labels": metric_labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=metric_labels).tolist(),
        "other_prediction_count": sum(x == "other" for x in y_pred),
        "mean_inference_ms": sum(r["inference_ms"] for r in rows) / len(rows),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"metrics-{payload.get('model_version', 'unknown')}-{payload.get('group', 'unknown')}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

