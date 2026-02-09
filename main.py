"""Entry point — run with:  uv run python main.py [--config config.yaml] [--mode train|eval] [--ckpt path]"""
import argparse

from src.config import load_config
from src.train import train
from src.eval import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "train":
        train(cfg)
    else:
        if not args.ckpt:
            raise SystemExit("--ckpt required for eval mode")
        results = evaluate(cfg, args.ckpt)
        for k, v in results.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
