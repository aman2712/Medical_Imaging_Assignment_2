# ENG2440 Assignment 2: Tumor Segmentation in MRI Volumes

Student: Md Aman Khan (25011138)

This project finds the whole tumor in 3D brain MRI scans (MSD Task01 BrainTumour) using a small 3D U-Net.

## Setup

1. Use Python 3.13 and install the packages: `pip install -r requirements.txt`
2. Download the dataset (`Task01_BrainTumour`) and the Moodle file `assignment2_split.csv`. Put both next to the notebook.
3. If the dataset is somewhere else, change `DATA_ROOT` in the Configuration cell of the notebook. This is the only path to change.

## Run order

1. Open `25011138_ENG2440_A2.ipynb`.
2. Run all cells from top to bottom.

On the first run the notebook does these steps itself:

- It saves a preprocessed copy of every case in `cache/`.
- It trains the two models and saves them in `checkpoints/`.
- It reports results on validation and test data, then the two bonus tasks.

If `checkpoints/` already holds the saved models, training is skipped and they are loaded instead.

## Reproducibility

- Seed: 42 for everything (model start, patch sampling, training).
- Hardware: Apple M4 Max, 36 GB memory, run on the Apple GPU (`mps`).
- Time: training takes about 17 minutes per model. Predicting the 73 test cases takes about 2 minutes. The other steps take a few minutes each.
- Results may differ slightly on other hardware.

## Files

- `25011138_ENG2440_A2.ipynb`: the executed notebook
- `scripts/`: helper code used by the notebook
- `assignment2_split.csv`: the supplied train/validation/test split
- `requirements.txt`: package versions
- `ai_use_declaration.txt`: AI use declaration
- `report.tex`, `report.pdf`: the report

The dataset, `cache/` and `checkpoints/` are not uploaded.
