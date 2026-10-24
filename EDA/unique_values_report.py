"""Per-feature unique-values report for manual review of a training CSV.

Usage:
    python unique_values_report.py <train.csv> [-o <output.csv>] [-n <max_values>]

Output CSV columns (rows sorted by n_unique, descending):
    feature       - column name
    n_unique      - number of distinct values (NULL counted as one value if present)
    n_null        - number of rows where the feature is missing
    unique_values - up to n values, ", " separated, most frequent first
                    (ties: value ascending); missing values are written as the
                    text NULL and ranked by their frequency like any other value
"""
import argparse
from pathlib import Path

import pandas as pd

NULL_TEXT = "NULL"


def format_value(v):
    # 3.0 -> "3" so integer-valued floats (columns with NaNs) read cleanly
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def summarize(df, max_values):
    rows = []
    for col in df.columns:
        s = df[col]
        n_null = int(s.isna().sum())
        counts = s.dropna().value_counts()

        # most frequent first; ties broken by value ascending so output is deterministic
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        shown = [(format_value(v), c) for v, c in ranked]
        if n_null:
            # NULL takes its place by frequency like any other value
            pos = next((i for i, (_, c) in enumerate(shown) if c < n_null), len(shown))
            shown.insert(pos, (NULL_TEXT, n_null))

        rows.append({
            "feature": col,
            "n_unique": len(shown),
            "n_null": n_null,
            "unique_values": ", ".join(v for v, _ in shown[:max_values]),
        })
    # stable sort keeps original column order among ties
    return pd.DataFrame(rows).sort_values("n_unique", ascending=False, kind="stable")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="training CSV to scan")
    parser.add_argument("-o", "--output", type=Path,
                        help="output CSV (default: <input>_unique_values.csv next to the input)")
    parser.add_argument("-n", "--max-values", type=int, default=10,
                        help="max unique values listed per feature (default: 10)")
    args = parser.parse_args()

    # "NA", "", "NaN" etc. are read as missing and reported as NULL
    df = pd.read_csv(args.csv)
    out = args.output or args.csv.with_name(f"{args.csv.stem}_unique_values.csv")
    summarize(df, args.max_values).to_csv(out, index=False)
    print(f"{len(df.columns)} features -> {out}")


if __name__ == "__main__":
    main()
