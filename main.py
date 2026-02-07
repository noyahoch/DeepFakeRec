from __future__ import annotations

import argparse

from src.config import load_config
from src.eval import evaluate
from src.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "train":
        train(cfg)
    else:
        if args.ckpt is None:
            raise SystemExit("--ckpt is required for eval mode")
        metrics = evaluate(cfg, args.ckpt)
        print(metrics)


if __name__ == "__main__":
    main()
