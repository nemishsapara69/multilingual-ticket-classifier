import argparse
import json
import os
import random
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def set_seed(seed: int, torch_module=None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch_module is not None:
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)


def load_params(params_path: Path) -> dict:
    with params_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def _metrics_from_predictions(labels, preds) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }


def _load_torch_module():
    try:
        import torch

        return torch
    except Exception:
        return None


def _train_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, model_dir: Path, metrics_path: Path) -> dict:
    model_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=800)),
        ]
    )

    pipeline.fit(train_df["text"], train_df["category"])

    val_preds = pipeline.predict(val_df["text"])
    test_preds = pipeline.predict(test_df["text"])

    val_metrics = _metrics_from_predictions(val_df["category"], val_preds)
    test_metrics = _metrics_from_predictions(test_df["category"], test_preds)

    dump(pipeline, model_dir / "baseline_pipeline.joblib")

    model_info = {
        "backend": "baseline",
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_dir": str(model_dir),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    return model_info


def _train_transformer(cfg: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, model_dir: Path, metrics_path: Path) -> dict:
    from datasets import Dataset
    from huggingface_hub import login
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    encoder = LabelEncoder()
    train_df["label"] = encoder.fit_transform(train_df["category"])
    val_df["label"] = encoder.transform(val_df["category"])
    test_df["label"] = encoder.transform(test_df["category"])

    label_list = list(encoder.classes_)
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=int(cfg["max_length"]))

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]]).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]]).map(tokenize, batched=True)

    train_ds = train_ds.remove_columns([col for col in train_ds.column_names if col not in {"input_ids", "attention_mask", "label"}])
    val_ds = val_ds.remove_columns([col for col in val_ds.column_names if col not in {"input_ids", "attention_mask", "label"}])
    test_ds = test_ds.remove_columns([col for col in test_ds.column_names if col not in {"input_ids", "attention_mask", "label"}])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label={str(k): v for k, v in id2label.items()},
        label2id=label2id,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir = model_dir.parent / "checkpoints"

    args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=int(cfg["batch_size"]),
        per_device_eval_batch_size=int(cfg["batch_size"]),
        num_train_epochs=int(cfg["num_train_epochs"]),
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        logging_steps=25,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    test_metrics = trainer.evaluate(eval_dataset=test_ds)

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    model_info = {
        "backend": "transformer",
        "labels": label_list,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_dir": str(model_dir),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    if os.getenv("PUSH_TO_HUB", "0") == "1":
        repo_id = os.getenv("HF_REPO_ID")
        hf_token = os.getenv("HF_TOKEN")
        if repo_id and hf_token:
            login(token=hf_token)
            model.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            print(f"Pushed model to Hugging Face Hub: {repo_id}")
        else:
            print("Skipping Hugging Face push: set HF_REPO_ID and HF_TOKEN.")

    return model_info


def train_model(data_dir: Path, model_dir: Path, params_path: Path, metrics_path: Path) -> None:
    cfg = load_params(params_path)["train"]
    torch_module = _load_torch_module()
    set_seed(int(cfg["seed"]), torch_module=torch_module)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    mlflow.set_experiment("multilingual-ticket-classifier")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    use_baseline = os.getenv("FORCE_BASELINE", "0") == "1" or torch_module is None

    with mlflow.start_run():
        mlflow.log_params(cfg)
        mlflow.log_param("backend", "baseline" if use_baseline else "transformer")

        if use_baseline:
            model_info = _train_baseline(train_df, val_df, test_df, model_dir, metrics_path)
            mlflow.log_metrics(
                {
                    "val_accuracy": float(model_info["val_metrics"].get("accuracy", 0.0)),
                    "val_f1_macro": float(model_info["val_metrics"].get("f1_macro", 0.0)),
                    "test_accuracy": float(model_info["test_metrics"].get("accuracy", 0.0)),
                    "test_f1_macro": float(model_info["test_metrics"].get("f1_macro", 0.0)),
                }
            )
            print("Torch is unavailable in this environment. Trained baseline TF-IDF + Logistic Regression model.")
        else:
            model_info = _train_transformer(cfg, train_df, val_df, test_df, model_dir, metrics_path)
            mlflow.log_metrics(
                {
                    "val_accuracy": float(model_info["val_metrics"].get("eval_accuracy", 0.0)),
                    "val_f1_macro": float(model_info["val_metrics"].get("eval_f1_macro", 0.0)),
                    "test_accuracy": float(model_info["test_metrics"].get("eval_accuracy", 0.0)),
                    "test_f1_macro": float(model_info["test_metrics"].get("eval_f1_macro", 0.0)),
                }
            )

        mlflow.log_artifact(str(metrics_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multilingual ticket classifier.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--metrics_path", type=Path, default=Path("models/metrics.json"))
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        params_path=args.params,
        metrics_path=args.metrics_path,
    )


if __name__ == "__main__":
    main()
