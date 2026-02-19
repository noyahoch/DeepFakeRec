"""Entry point — run with:  uv run python main.py [--config run_configs/config.yaml] [--mode train|eval] [--ckpt path]"""

import argparse

from src.config import load_config
from src.eval import evaluate
from src.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="run_configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for logs, checkpoints, and wandb",
    )
    parser.add_argument(
        "--override",
        type=str,
        default=None,
        metavar="YAML",
        help="Optional override config (inherits from --config, only set keys to override).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, override_path=args.override)

    if args.mode == "train":
        train(cfg, run_name=args.run_name)
    else:
        if not args.ckpt:
            raise SystemExit("--ckpt required for eval mode")
        results = evaluate(cfg, args.ckpt)
        for k, v in results.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
