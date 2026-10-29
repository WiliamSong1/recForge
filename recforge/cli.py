import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="RecForge reproducible MovieLens experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--data", type=Path, default=Path("data/ml-1m"))
    train = sub.add_parser("train")
    train.add_argument("--data", type=Path, default=Path("data/ml-1m"))
    train.add_argument("--config", type=Path, default=Path("configs/default.json"))
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--tracking-uri")
    train.add_argument("--final-test", action="store_true", help="Explicitly unlock test evaluation after freezing configuration")
    tuning = sub.add_parser("tune")
    tuning.add_argument("--data", type=Path, default=Path("data/ml-1m"))
    tuning.add_argument("--config", type=Path, default=Path("configs/default.json"))
    tuning.add_argument("--trials", type=int, default=5)
    tuning.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        from .data import download as fetch
        print(fetch(args.data))
    elif args.command == "tune":
        from .tuning import tune
        tune(args.data, json.loads(args.config.read_text()), args.trials, args.output)
    else:
        from .pipeline import run
        run(args.data, args.output, json.loads(args.config.read_text()), args.tracking_uri, args.final_test)


if __name__ == "__main__":
    main()
