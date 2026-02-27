"""
CLI script: Generate synthetic Telco Churn dataset.

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --samples 5000 --seed 99
"""

import argparse
import sys
from pathlib import Path

# Allow importing from src/ when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import generate_telco_data

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "telco_churn.csv"


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Telco Churn dataset.")
    parser.add_argument("--samples", type=int, default=10000, help="Number of rows (default: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"Generating {args.samples:,} customer records (seed={args.seed})...")
    df = generate_telco_data(n_samples=args.samples, random_state=args.seed)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"Saved → {OUTPUT_PATH}")
    print(f"Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Churn : {churn_rate:.1%} ({(df['Churn']=='Yes').sum():,} churners)")


if __name__ == "__main__":
    main()
