# Data setup — ASVspoof LA datasets

This guide explains how to download and arrange ASVspoof **Logical Access (LA)** data so paths match `run_configs/config.yaml` and `src/dataset.py`.

**Project root:** All commands assume you are in the **project root** (directory containing `run_configs/config.yaml`, `main.py`, and `data/`). Run `cd <project_root>` before each step if needed.

---

## LA_2019 — download and extract in `data/LA_2019/`

Everything for ASVspoof 2019 LA goes in `data/LA_2019/`. Default `run_configs/config.yaml` points here for training and eval.

### Create folder and download into `data/LA_2019/`

- **URL:**  
  https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y
- **Size:** ~7.1 GB (zip).

**Browser:** Create `data/LA_2019/` if needed, then save as `data/LA_2019/ASVspoof2019_LA.zip`.

**Command line:**

```bash
cd <project_root>
mkdir -p data/LA_2019
wget -O data/LA_2019/ASVspoof2019_LA.zip "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"
```

**Optional checksum (DataShare README):**  
MD5: `30c98f11d8b2bc21f2c257bfd78bb5c5` → `md5sum data/LA_2019/ASVspoof2019_LA.zip`

### Extract in `data/LA_2019/`

```bash
cd <project_root>/data/LA_2019
unzip ASVspoof2019_LA.zip
cd ../..
```

After extraction, `data/LA_2019/` should contain:

- `ASVspoof2019_LA_train/` — training audio (e.g. `flac/` inside)
- `ASVspoof2019_LA_dev/` — development audio
- `ASVspoof2019_LA_eval/` — evaluation audio
- `ASVspoof2019_LA_cm_protocols/` — CM protocol files (`.trn`, `.trl`, etc.)
- `README.LA.txt` (if present)

Protocol format is space-separated: `speaker_id file_id - - key` with `key` in `bonafide` / `spoof`. `src/dataset.py` expects audio as `file_id.flac` under the relevant `flac/` directory.

---

## LA_2021 — download and extract in `data/LA_2021/`

Everything for ASVspoof 2021 LA eval goes in `data/LA_2021/`. Eval-only (no train/dev). Structure is assumed similar to LA_2019; adjust paths or parsing if your layout differs.

### Create folder and download into `data/LA_2021/`

**Audio (Zenodo)**  
- URL: https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz?download=1  
- Size: ~7.8 GB (tar.gz)

**Keys/labels (ASVspoof; required for ground truth)**  
- URL: https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz  

**Command line:**

```bash
cd <project_root>
mkdir -p data/LA_2021
wget -O data/LA_2021/ASVspoof2021_LA_eval.tar.gz "https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz?download=1"
wget -O data/LA_2021/ASVspoof2021_LA_keys.tar.gz "https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz"
```

**Optional checksum (Zenodo, eval archive):**  
MD5: `2abee34d8b0b91159555fc4f016e4562` → `md5sum data/LA_2021/ASVspoof2021_LA_eval.tar.gz`

### Extract in `data/LA_2021/`

```bash
cd <project_root>/data/LA_2021
tar -xzf ASVspoof2021_LA_eval.tar.gz
tar -xzf ASVspoof2021_LA_keys.tar.gz
cd ../..
```

After extraction you should have (paths may vary by archive layout):

- Eval audio, e.g. `data/LA_2021/ASVspoof2021_LA_eval/flac/`
- Protocol/key files, e.g. in a `keys/` or `cm_protocols/`-style directory

If LA_2021 key format matches LA_2019, set in `run_configs/config.yaml`:

```yaml
data:
  eval_protocol_path: "data/LA_2021/ASVspoof2021_LA_cm_protocols/ASVspoof2021_LA_eval.trl.txt"  # adjust to actual path
  eval_audio_dir: "data/LA_2021/ASVspoof2021_LA_eval/flac"  # adjust to actual path
```

If the structure or key format differs, update these paths and/or extend `parse_protocol()` in `src/dataset.py`.
