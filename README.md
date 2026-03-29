# DeepFakeRec

Audio deepfake detection with XLS-R and an SLS classifier.

## Install

```bash
pip install -r requirements.txt
```

## Before Running

The default config expects:

- ASVspoof 2019 LA training data under `data/LA_2019/`
- ASVspoof 2021 LA eval data under `data/LA_2021/`
- the XLS-R model at `models/wav2vec2-xls-r-300m`

Dataset layout details are in `DATA_SETUP.md`.

## Train

```bash
python main.py --config run_configs/config.yaml --mode train --run-name my_run
```

Checkpoints are saved in `checkpoints/`.

## Eval

```bash
python main.py --config run_configs/config.yaml --mode eval --ckpt checkpoints/<best.ckpt>
```

## Notes

- If your data or model paths are different, edit `run_configs/config.yaml` or use `--override path/to/override.yaml`.
- If you do not want Weights & Biases logging, set `logging.use_wandb: false` in the config.
