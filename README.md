# EEG motor-imagery classification

A notebook comparing CSP + LDA with a compact EEGNet model on BCI Competition IV
2a. The task is to recognise four imagined movements: left hand, right hand, feet,
and tongue. Each subject has a separate model trained on session T and evaluated
on session E.

The project uses local MATLAB files. It runs in the existing `neuroAI` conda
environment and does not download data during training.

## Files

```text
motor_imagery.ipynb                  # Loading, plots, both models, evaluation
REPORT.md                           # Explanation and reading material
REPORT.html                         # The same explanation for a web browser
data_sources.json                   # Official E-file URLs and SHA-256 hashes
BCICIV 2a Motor Imagery EEG Dataset/  # A01T.mat ... A09E.mat; ignored by Git
results/                            # One folder per experiment
```

## Run in VS Code

1. Open this project folder with **File > Open Folder**.
2. Open `motor_imagery.ipynb`. VS Code needs the Microsoft Python and Jupyter
   extensions to work with the notebook.
3. Use **Select Kernel > Python Environments** and choose `neuroAI`. On this
   machine its interpreter is `C:\Users\muzz\anaconda3\envs\neuroAI\python.exe`.
4. Run the import cell and check the printed device. CUDA uses the available GPU;
   CPU also works, but training will take longer.
5. Run the settings cell, then continue from top to bottom. The default is
   `SUBJECTS = [1]`. Use `SUBJECTS = list(range(1, 10))` for all nine people.

If the data folder cannot be found, set `PROJECT_DIR` in the settings cell to
the project folder's absolute path. Restart the kernel and run all cells again
after changing the preprocessing or model settings.

[VS Code's notebook guide](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
explains the kernel picker and cell controls.

## Run with Anaconda / Jupyter

Open Anaconda Prompt and run:

```bat
conda activate neuroAI
cd /d "C:\Users\muzz\Documents\Research Work\EEG motor-imagery classification"
python -m jupyterlab
```

JupyterLab is already present in `neuroAI` on this machine. Open
`motor_imagery.ipynb` in the browser and select the `neuroAI` kernel. If
Jupyter does not list that kernel, register the existing environment once:

```bat
conda activate neuroAI
python -m ipykernel install --user --name neuroAI --display-name "Python (neuroAI)"
```

This registers a kernel; it does not create another environment or install the
project's libraries. The standalone Notebook launcher is not installed in this
environment; JupyterLab opens and runs the same `.ipynb` file. In Anaconda
Navigator, select `neuroAI` before launching JupyterLab.

## What the notebook does

The loader skips the eye-calibration blocks, keeps 22 EEG channels, converts
microvolts to volts, and applies an 8–30 Hz Butterworth filter to each continuous
imagery run. It extracts 0.5–3.5 seconds after the cue, including the endpoint.
At 250 Hz, each trial has 751 time samples. Dataset-marked artifact trials are
excluded by default; the output audit records the counts.

CSP learns six spatial log-power features, which shrinkage LDA classifies.
EEGNet learns temporal and spatial filters directly from the trial arrays. It
uses training runs 1–5 to fit and run 6 to select the epoch count, then trains a
fresh model on all T trials for that many epochs. Session E does not select
settings, normalization statistics, or stopping time.

The final scores measure offline prediction of another recording session from
the same person. They do not measure generalisation to new people or implement
the competition's original continuous, causal evaluation.

## Outputs

The verified full run is documented in [VERIFICATION.md](VERIFICATION.md). Mean accuracy was 62.0% for CSP + LDA and 54.2% for EEGNet, with the settings described below.

Each experiment saves a dated folder under `results/` containing:

- `metrics.csv` and `summary.csv`: per-subject accuracy/kappa and subject averages.
- `per_subject_comparison.csv`: both methods side by side.
- `predictions.csv` and trial-index files: evaluation labels, predictions, and
  original trial identities. Classes are numbered 0–3 in the documented order.
- `preprocessing_audit.csv`: total, flagged, and retained trials in each run.
- `config.json`, `input_files.json`, and `model_selection.csv`: settings, package
  versions, file hashes, selected epochs, and validation scores.
- `plots/`: signal preview, comparison chart, confusion matrices, and EEGNet curves.
- `histories/`: selection-stage loss and accuracy by epoch.
- `models/`: final EEGNet weights with normalization statistics.

Training curves belong to the model-selection stage. The model used for E is
the fresh all-T refit. `summary.csv` gives each subject equal weight. Pooled
confusion matrices combine individual trial counts.

## Before uploading to GitHub

The `.gitignore` excludes MATLAB/GDF data, the original archive, checkpoints,
notebook caches, and local editor settings. It keeps result tables and plots
available for the repository. Clear notebook outputs before committing if they
contain local paths or results you do not want to publish. Never paste EEG arrays
into a notebook output: the data-file ignore rules do not cover embedded arrays.

Preview what Git would include:

```bat
git status --short --untracked-files=all
git check-ignore "BCICIV 2a Motor Imagery EEG Dataset/A01T.mat" archive.zip
```

Ignore rules do not remove files that Git already tracks. Review the staged
files before committing. No Git repository or remote is created by this project.

## Data and references

The nine T files came from the local dataset supplied for this project. The E
files were obtained from the [official BNCI catalog](https://bnci-horizon-2020.eu/database/data-sets),
which links to the Graz University of Technology mirror. These MAT files contain
the released E labels. The catalog lists the data under CC BY-ND 4.0; cite the
dataset authors and associated publication when reporting results. This
repository keeps dataset files out of Git.

Read [REPORT.md](REPORT.md) for the EEG explanation, evaluation limits, and a
reading path through MNE, CSP, and EEGNet. The main technical references are the
[dataset description](https://www.bbci.de/competition/iv/desc_2a.pdf),
[competition review](https://pmc.ncbi.nlm.nih.gov/articles/PMC3396284/), and
[EEGNet paper](https://arxiv.org/abs/1611.08024).
