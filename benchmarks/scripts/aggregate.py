"""Aggregate independent, predeclared training seeds without pooling users."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def aggregate(paths):
    reports = [json.loads(Path(p).read_text()) for p in paths]
    if len({r["config"]["seed"] for r in reports}) != len(reports):
        raise ValueError("Expected distinct training seeds")
    reference = {k: v for k, v in reports[0]["config"].items() if k != "seed"}
    for report in reports:
        assert {k: v for k, v in report["config"].items() if k != "seed"} == reference
        assert report["dataset"] == reports[0]["dataset"] and report["split"] == reports[0]["split"]
    quality = pd.DataFrame([dict(row, seed=r["config"]["seed"]) for r in reports for row in r["quality"]])
    summary = quality.groupby(["cohort", "model"])["ndcg@10"].agg(["mean", "std", "count"]).reset_index()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/seed_summary.csv"))
    args = parser.parse_args()
    table = aggregate(args.reports)
    table.to_csv(args.output, index=False)
    print(table.to_string(index=False))
