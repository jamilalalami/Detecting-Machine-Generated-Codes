import argparse
import os
import joblib
import torch
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------
# Utility: model loader
# -----------------------
def load_model(model_type, checkpoint_path, num_labels):
    if model_type == "baseline":
        model = joblib.load(checkpoint_path)
        return None, model

    elif model_type in ["transformer", "hybrid", "multitask"]:
        tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(checkpoint_path))
        model = AutoModelForSequenceClassification.from_pretrained(
            os.path.dirname(checkpoint_path),
            num_labels=num_labels
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        return tokenizer, model

    else:
        raise ValueError(f"Unknown model type: {model_type}")

# -----------------------
# Utility: run evaluation
# -----------------------
def evaluate(model_type, checkpoint, subtask, data_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    codes = df["code"].tolist()
    y_true = df["label"].tolist()
    num_labels = len(set(y_true))

    tokenizer, model = load_model(model_type, checkpoint, num_labels)

    # -----------------------
    # SVM baseline evaluation
    # -----------------------
    if model_type == "baseline":
        y_pred = model.predict(codes)

    # -----------------------
    # Transformer inference
    # -----------------------
    else:
        y_pred = []
        for text in codes:
            inputs = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
                y_pred.append(pred)

    # -----------------------
    # Metrics
    # -----------------------
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    # Save results
    with open(f"{output_dir}/eval_report_{subtask}.txt", "w") as f:
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write(report)

    np.savetxt(f"{output_dir}/confusion_matrix_{subtask}.csv", cm, delimiter=",")

    print("\n========== Evaluation Summary ==========")
    print(f"Subtask: {subtask}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(report)
    print("Confusion matrix saved.")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-type", required=True,
                        choices=["baseline", "transformer", "hybrid", "multitask"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subtask", required=True, choices=["A", "B", "C"])
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="reports/evaluations/")

    args = parser.parse_args()

    evaluate(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        subtask=args.subtask,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
