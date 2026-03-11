# How to run DeepFakeRec (with fixes)

Step-by-step checklist to train and evaluate the Audio Deepfake Detection model (XLS-R + SLS).

---

## 1. Project root

All commands assume you are in the project root (the directory that contains `main.py`, `run_configs/`, and `data/`).

---

## 2. Environment

- Use the project’s Python environment (e.g. **uv**: `uv run python main.py ...`, or activate the venv that has the dependencies).
---

## 3. Data (ASVspoof 2019 LA)

Data must be under `data/LA_2019/` as described in **DATA_SETUP.md**.

- **Download** (from project root):

  ```bash
  mkdir -p data/LA_2019
  wget -O data/LA_2019/ASVspoof2019_LA.zip "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"
  ```

- **Extract**:

  ```bash
  cd data/LA_2019
  unzip ASVspoof2019_LA.zip
  cd ../..   # back to project root
  ```

After extraction you should have:

- `ASVspoof2019_LA_train/flac/`, `ASVspoof2019_LA_eval/flac/`
- `ASVspoof2019_LA_cm_protocols/` with `ASVspoof2019.LA.cm.train.trn.txt` and `ASVspoof2019.LA.cm.eval.trl.txt`

---

## 4. XLS-R model

The default config uses a **local** path: `models/wav2vec2-xls-r-300m`.

- If that directory does not exist, either:
  - Download/copy the HuggingFace `facebook/wav2vec2-xls-r-300m` model into `models/wav2vec2-xls-r-300m`, or
  - Set `model.pretrained_name` in config (or in an override YAML) to your model path (e.g. `facebook/wav2vec2-xls-r-300m` or a local path).

---

## 5. Training

```bash
uv run python main.py --config run_configs/config.yaml --mode train --run-name my_run
```

- **Optional:** use an override YAML for data paths, model path, etc.:  
  `--override path/to/override.yaml`
- Checkpoints are written to `checkpoints/` (or as in config); the best by validation EER is kept.

---

## 6. Evaluation

```bash
uv run python main.py --config run_configs/config.yaml --mode eval --ckpt checkpoints/best-epoch=X-val_eer=Y.YYYY.ckpt
```

- Replace `best-epoch=X-val_eer=Y.YYYY.ckpt` with your actual best checkpoint filename.
- To write scores to a file, set `eval.save_scores_path` in the config (or in the override YAML) to the desired output path.

---

## 7. Eval on ASVspoof 2021 DF (optional)

If you have the 2021 DF dataset and protocol:

- Add an override YAML that sets:
  - `data.eval_protocol_path` → path to the 2021 DF protocol file
  - `data.eval_audio_dir` → path to the 2021 DF eval audio directory
- Run the same eval command with:  
  `--override path/to/df2021_override.yaml`

---

---

## Summary

| Step | Action |
|------|--------|
| 1 | From project root (directory with `main.py`, `run_configs/`, `data/`) |
| 2 | Use project env (`uv run` or venv) |
| 3 | Put ASVspoof 2019 LA in `data/LA_2019/` (see DATA_SETUP.md) |
| 4 | Ensure XLS-R model at `models/wav2vec2-xls-r-300m` or set in config |
| 5 | Train: `uv run python main.py --config run_configs/config.yaml --mode train --run-name my_run` |
| 6 | Eval: `uv run python main.py --config run_configs/config.yaml --mode eval --ckpt <best.ckpt>` |
| 7 | (Optional) Eval on 2021 LA: `--override run_configs/eval_la21_full.yaml` |
