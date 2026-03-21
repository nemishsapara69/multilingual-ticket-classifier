import argparse
import json
from pathlib import Path

import pandas as pd
import requests

from src.inference import InferenceEngine


def run_api_prediction(text: str, api_url: str, timeout: int) -> str:
    response = requests.post(api_url, json={"text": text}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return str(payload["category"]).strip()


def run_local_prediction(text: str, model_dir: str) -> str:
    engine = run_local_prediction.engine
    if engine is None:
        run_local_prediction.engine = InferenceEngine.from_local_or_hub(local_model_dir=model_dir)
        engine = run_local_prediction.engine
    pred = engine.predict(text)
    return str(pred["category"]).strip()


run_local_prediction.engine = None


def evaluate(df: pd.DataFrame, mode: str, api_url: str, model_dir: str, timeout: int) -> dict:
    predictions = []

    for _, row in df.iterrows():
        text = str(row["text"])
        if mode == "api":
            pred = run_api_prediction(text=text, api_url=api_url, timeout=timeout)
        else:
            pred = run_local_prediction(text=text, model_dir=model_dir)
        predictions.append(pred)

    out = df.copy()
    out["predicted_category"] = predictions
    out["expected_category"] = out["expected_category"].astype(str).str.strip()
    out["is_correct"] = out["predicted_category"] == out["expected_category"]

    overall_accuracy = float(out["is_correct"].mean())

    per_language = (
        out.groupby("language")["is_correct"]
        .mean()
        .sort_index()
        .to_dict()
    )
    per_language = {k: float(v) for k, v in per_language.items()}

    per_category = (
        out.groupby("expected_category")["is_correct"]
        .mean()
        .sort_index()
        .to_dict()
    )
    per_category = {k: float(v) for k, v in per_category.items()}

    mistakes = out[~out["is_correct"]][["language", "text", "expected_category", "predicted_category"]]

    return {
        "num_samples": int(len(out)),
        "overall_accuracy": overall_accuracy,
        "accuracy_by_language": per_language,
        "accuracy_by_expected_category": per_category,
        "num_errors": int((~out["is_correct"]).sum()),
        "errors": mistakes.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multilingual ticket classifier.")
    parser.add_argument("--input", type=Path, default=Path("data/eval/eval_queries.csv"))
    parser.add_argument("--mode", choices=["api", "local"], default="api")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8000/predict")
    parser.add_argument("--model_dir", type=str, default="models/best")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("reports/eval_report.json"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required_cols = {"language", "text", "expected_category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    report = evaluate(
        df=df,
        mode=args.mode,
        api_url=args.api_url,
        model_dir=args.model_dir,
        timeout=args.timeout,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Samples: {report['num_samples']}")
    print(f"Overall accuracy: {report['overall_accuracy']:.4f}")
    print("Accuracy by language:")
    for lang, acc in report["accuracy_by_language"].items():
        print(f"  {lang}: {acc:.4f}")
    print("Accuracy by expected category:")
    for cat, acc in report["accuracy_by_expected_category"].items():
        print(f"  {cat}: {acc:.4f}")
    print(f"Errors: {report['num_errors']}")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
