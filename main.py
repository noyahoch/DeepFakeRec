"""Entry point — run with:  uv run python main.py [--config run_configs/config.yaml] [--mode train|eval] [--ckpt path]"""

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="run_configs/config.yaml", help="Path to config YAML"
    )
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA (force CPU-only mode).",
    )
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
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Override model.freeze_backbone to False (so backbone trains). Without this, both runs use the same config → bit-identical results.",
    )
    parser.add_argument(
        "--wandb-key",
        dest="wandb_key",
        type=str,
        default=None,
        help="Path to file containing wandb API key for login.",
    )
    args = parser.parse_args()

    if args.no_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from src.config import load_config
    from src.eval import evaluate
    from src.train import train

    cfg = load_config(args.config, override_path=args.override)
    if args.unfreeze_backbone:
        cfg.model.freeze_backbone = False

    if args.mode == "train":
        train(cfg, run_name=args.run_name, wandb_key_path=args.wandb_key)
    else:
        if not args.ckpt:
            raise SystemExit("--ckpt required for eval mode")
        results = evaluate(cfg, args.ckpt, run_name=args.run_name)
        for k, v in results.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
