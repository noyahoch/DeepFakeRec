# ASVspoof 2019 LA — Download and Data Setup

This guide explains how to download the ASVspoof 2019 **Logical Access (LA)** dataset and arrange it so it matches the paths used in `config.yaml` and `src/dataset.py`.

**Project root:** All commands below assume you are in the **project root**—the directory that contains `config.yaml`, `main.py`, and the `data/` folder. Before each step, run `cd <project_root>` (replace `<project_root>` with the path to your repo).

---

## 1. Download the LA dataset

- **URL:**  
  https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y

- **Size:** ~7.1 GB (zip).

- **Options:**
  - **Browser:** Open the URL in a browser; you may need to accept the site’s terms to start the download. If you do, save the file as `LA.zip` inside the project’s `data/` folder (create `data/` if needed).
  - **Command line:** Checkpoint the project directory, then run:
    ```bash
    cd <project_root>
    mkdir -p data
    wget -O data/LA.zip "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"
    ```
  - **Optional checksum (README on DataShare):**  
    MD5 for `LA.zip`: `30c98f11d8b2bc21f2c257bfd78bb5c5`  
    From the project root: `md5sum data/LA.zip`

---

## 2. Extract the zip

Checkpoint the project directory, then run:

```bash
cd <project_root>
cd data
unzip LA.zip -d LA_2019
cd ..
```

The zip is extracted into `data/LA_2019/`. Inside it you should see something like:

- `LA_2019/ASVspoof2019_LA_train/` — training audio (e.g. `flac/` inside)
- `LA_2019/ASVspoof2019_LA_dev/` — development audio
- `LA_2019/ASVspoof2019_LA_eval/` — evaluation audio
- `LA_2019/ASVspoof2019_LA_cm_protocols/` — countermeasure (CM) protocol files
- `LA_2019/README.LA.txt` (if present)
