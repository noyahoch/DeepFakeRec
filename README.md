# DeepFakeRec

Audio deepfake detection with XLS-R and an SLS classifier.

## Install

```bash
pip install -r requirements.txt
```

## Quick Run On The Included Samples

This repository includes a very small `sample_data/` folder so the project can be run immediately after install.

The sample data contains only a few files from:

- the training split
- the evaluation split

For this sample config, the XLS-R backbone is not set explicitly in the YAML, so it will be downloaded automatically on first run.

Use the sample config for a quick test:

```bash
python main.py --config run_configs/run_on_samples_config.yaml --mode train --run-name sample_run --no-cuda
python main.py --config run_configs/run_on_samples_config.yaml --mode eval --ckpt checkpoints/<best.ckpt> --no-cuda
```

Checkpoints are saved in `checkpoints/`.

## Full Dataset

If you want to train or evaluate on the full ASVspoof datasets, download the real data and place it under:

- `data/LA_2019/`
- `data/LA_2021/`

The full setup is described in `DATA_SETUP.md`.

Then run with the full config:

```bash
python main.py --config run_configs/config.yaml --mode train --run-name my_run
python main.py --config run_configs/config.yaml --mode eval --ckpt checkpoints/<best.ckpt>
```

## Notes

- The full config expects the XLS-R model at `models/wav2vec2-xls-r-300m`.
- If your data or model paths are different, edit the config file or use `--override path/to/override.yaml`.
- If you do not want Weights & Biases logging, set `logging.use_wandb: false` in the config.
