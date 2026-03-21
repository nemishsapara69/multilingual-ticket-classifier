import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TEXT_CANDIDATES = ["text", "message", "ticket_text", "description", "body"]
LABEL_CANDIDATES = ["category", "label", "ticket_category", "intent"]


def _find_column(columns: list[str], candidates: list[str]) -> str:
    lookup = {c.lower().strip(): c for c in columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    raise ValueError(f"Could not find any of these columns: {candidates}. Found: {columns}")


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)

    # Remove emojis and pictographs while preserving multilingual letters.
    text = re.sub(r"[\U00010000-\U0010FFFF]", " ", text)

    # Remove control characters and normalize whitespace.
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_raw_dataset(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path)
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    raise ValueError("Supported formats: .csv, .xlsx, .xls")


def preprocess(input_path: Path, output_dir: Path, test_size: float = 0.15, val_size: float = 0.15) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_dataset(input_path)

    text_col = _find_column(list(raw_df.columns), TEXT_CANDIDATES)
    label_col = _find_column(list(raw_df.columns), LABEL_CANDIDATES)

    df = raw_df[[text_col, label_col]].copy()
    df.columns = ["text", "category"]

    df["text"] = df["text"].astype(str).map(clean_text)
    df["category"] = df["category"].astype(str).str.strip()

    df = df[(df["text"] != "") & (df["category"] != "")].dropna().reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df,
        test_size=(test_size + val_size),
        random_state=42,
        stratify=df["category"],
    )

    relative_val = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - relative_val),
        random_state=42,
        stratify=temp_df["category"],
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    metadata = {
        "source_file": str(input_path),
        "num_rows": len(df),
        "num_train": len(train_df),
        "num_val": len(val_df),
        "num_test": len(test_df),
        "categories": sorted(df["category"].unique().tolist()),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess multilingual support ticket data.")
    parser.add_argument("--input", required=True, type=Path, help="Path to raw CSV/XLSX dataset.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory for processed outputs.")
    args = parser.parse_args()

    preprocess(input_path=args.input, output_dir=args.output_dir)
    print(f"Processed dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
