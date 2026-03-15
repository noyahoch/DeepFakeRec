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

---

## Eval metadata from trial metadata (LA_2021)

For LA_2021, create `eval_metadata.txt` from the keys `trial_metadata.txt` by keeping only lines that end with `eval`:

```bash
cd <project_root>
grep 'eval$' data/LA_2021/keys/LA/CM/trial_metadata.txt > data/LA_2021/keys/LA/CM/eval_metadata.txt
```

---

## In-The-Wild — download and prepare in `data/in_the_wild/`

This repo expects an ASVspoof-style protocol file, so for In-The-Wild you need one extra preparation step after download.

### 1. Create the target folder

```bash
cd <project_root>
mkdir -p data/in_the_wild
```

### 2. Download the dataset archive

Official release page:

- https://huggingface.co/datasets/mueller91/In-The-Wild

Download the archive `release_in_the_wild.zip` into `data/in_the_wild/`.

If you prefer the browser, save it as:

- `data/in_the_wild/release_in_the_wild.zip`

### 3. Extract the archive

```bash
cd <project_root>/data/in_the_wild
unzip release_in_the_wild.zip
cd ../..
```

After extraction you should have a folder like:

- `data/in_the_wild/release_in_the_wild/`

Inside it you should see the audio files plus the metadata CSV shipped with the dataset, usually:

- `data/in_the_wild/release_in_the_wild/meta.csv`

### 4. Convert the metadata CSV into this repo's protocol format

```bash
cd <project_root>
uv run python scripts/prepare_in_the_wild_protocol.py \
  --metadata data/in_the_wild/release_in_the_wild/meta.csv \
  --output data/in_the_wild/protocol.txt
```

That generates:

- `data/in_the_wild/protocol.txt`

### 5. Run evaluation with the override config

```bash
uv run python main.py \
  --config run_configs/config.yaml \
  --mode eval \
  --ckpt checkpoints/<your-best-checkpoint>.ckpt \
  --override run_configs/eval_in_the_wild.yaml
```
