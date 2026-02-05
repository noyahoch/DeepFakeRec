# DeepFakeRec
Here is a comprehensive `README.md` style TODO list to guide your implementation of the paper. This breaks down the project into logical stages, from setting up the environment to calculating the final metrics.

---

# TODO: Audio Deepfake Detection (XLS-R + SLS)

This document outlines the roadmap for implementing the paper *"Audio Deepfake Detection with Self-supervised XLS-R and SLS classifier"*. The core objective is to build a detector that utilizes the intermediate representations of a pre-trained XLS-R model via a Sensitive Layer Selection (SLS) mechanism.

## 1. Environment & Prerequisites

* [ ] **Initialize Project Repository**
* Create a clean directory structure (e.g., `/src`, `/data`, `/checkpoints`).
* Initialize git.


* [ ] **Install Dependencies**
* [ ] `torch` (PyTorch for deep learning).
* [ ] `transformers` (HuggingFace for loading `wav2vec2-xls-r-300m`).
* [ ] `librosa` or `torchaudio` (For audio loading/processing).
* [ ] `numpy` & `pandas` (For data manipulation).
* [ ] `scikit-learn` (For calculating EER/metrics).



## 2. Data Preparation (ASVspoof)

The paper primarily uses ASVspoof 2019 (LA) for training and ASVspoof 2021 (DF/LA) for evaluation.

* [ ] **Dataset Acquisition**
* [ ] Download ASVspoof 2019 Logical Access (LA) dataset.
* [ ] Download ASVspoof 2021 Deepfake (DF) dataset.


* [ ] **Data Parsers (`dataset.py`)**
* [ ] Implement a function to parse ASVspoof `protocol.txt` files (mapping filenames to `bonafide`/`spoof` labels).
* [ ] Create a PyTorch `Dataset` class:
* [ ] Load audio files.
* [ ] **Fixed Length Processing:** Implement truncation (cropping) or padding (looping/zero-pad) to ensure a fixed input size (typically 4 seconds / ~64k samples).




* [ ] **Data Augmentation (RawBoost)**
* [ ] Implement `RawBoost.py` (or port from standard repositories).
* [ ] Add Linear and Non-Linear Convolutional Noise.
* [ ] Add Impulsive Signal Noise.
* [ ] **Integration:** Integrate RawBoost into the `Dataset` class (apply only during the training phase).



## 3. Model Architecture

Implementation of the core SSL backbone and the novel SLS module.

* [ ] **Backbone Loading**
* [ ] Load `facebook/wav2vec2-xls-r-300m` using HuggingFace.
* [ ] **Critical:** Configure the model to return *hidden states* for all layers (`output_hidden_states=True`), not just the final embedding.
* [ ] Implement a "Freezing" switch to freeze XLS-R weights initially (optional but recommended for stability).


* [ ] **Sensitive Layer Selection (SLS) Module**
* [ ] Create a learnable weight parameter vector of size  (where  typically: 24 layers + 1 embedding).
* [ ] Implement the `forward` pass:
* Stack hidden states from the backbone.
* Normalize weights (e.g., via Softmax) for stability.
* Compute the weighted sum of hidden states: .




* [ ] **Classification Head**
* [ ] Implement Max Pooling (aggregate over the time dimension).
* [ ] Add the Fully Connected (FC) layers.
* [ ] Add `SELU` activation (as specified in the paper/repo).
* [ ] Add the final projection layer to 2 classes (Real vs. Fake).



## 4. Training Pipeline

* [ ] **Loss Function**
* [ ] Implement **Weighted Cross Entropy Loss**.
* [ ] Calculate class weights based on the training set balance (Total Samples / Class Samples) to handle the imbalance between bonafide and spoof data.


* [ ] **Optimization**
* [ ] Setup `Adam` or `AdamW` optimizer (Learning rate ~ for backbone, ~ for head).
* [ ] (Optional) Implement a learning rate scheduler (Cosine Annealing is common).


* [ ] **Training Loop (`train.py`)**
* [ ] Create the training loop (Forward -> Loss -> Backward -> Step).
* [ ] Implement `Gradient Accumulation` (XLS-R is memory heavy; you might need to accumulate gradients to simulate a larger batch size).
* [ ] Add logging (Loss per epoch, Accuracy).
* [ ] Save model checkpoints (e.g., "best_loss.pth").



## 5. Evaluation & Inference

* [ ] **Inference Script (`eval.py`)**
* [ ] Load the trained checkpoint.
* [ ] Run inference on the ASVspoof 2021 DF evaluation set.
* [ ] Output a score file (filename, score) for every trial.


* [ ] **Metrics**
* [ ] Implement **EER (Equal Error Rate)** calculation.
* [ ] Implement **min-tDCF** (standard ASVspoof metric) using the official `tDCF_python` calculator provided by the ASVspoof organizers.



## 6. Verification

* [ ] **Sanity Check**
* [ ] Train on a tiny subset (e.g., 100 files) to ensure loss decreases.
* [ ] Check if the SLS weights change from their initialization (ensures the layer selection is actually learning).


* [ ] **Full Run**
* [ ] Train on full ASVspoof 2019 LA Train partition.
* [ ] Validate on ASVspoof 2019 LA Dev partition.
* [ ] Test on ASVspoof 2021 DF Eval partition.