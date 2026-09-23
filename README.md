# CNN_evaluation

## Purpose

Trains and evaluates VGG16-based convolutional neural networks that predict human emotional
valence ratings for IAPS (International Affective Picture System) images, framed both as a
classification problem (`train.py`) and as a regression problem (`train_regression.py`,
`test_regression.py`). In addition to a standard VGG backbone, `network.py` defines a custom
dual-pathway **"Amygdala"** architecture (`Amygdala`, `Amygdala_lowroad`) intended to mimic a
fast subcortical ("low road", lateral + central nucleus) route alongside a slower cortical
("high road") route to emotional response, plus plain VGG regression heads (`VGGReg`,
`VGGbnReg`).

This repository contains only the model/training/evaluation source code. It does **not**
include any IAPS images, CSV rating files, or trained model checkpoints - those must be
supplied separately (see How to Use and Dependencies below).

## Contents

- `network.py` — model definitions: `VGG`, `VGGReg`, `VGGbnReg`, `Amygdala`,
  `Amygdala_lowroad` (the "LA+CE" dual-pathway model), plus supporting helpers
  (`FeatureExtractor`, `GaussianSmoothing`, `vgg16_ori`/`vgg16_bn_ori`).
- `dataloader.py` — training-time dataset/dataloader construction: `cls_dataloader`
  (classification, `torchvision.datasets.ImageFolder`-style class-subfolder layout, with
  `ImbalancedDatasetSampler` on the training split) and `reg_dataloader` (regression, a
  CSV-driven `RegressionDataset` for train/val/test splits).
- `_dataloader.py` — the eval/test-only counterpart to `dataloader.py`: `reg_dataloader_test`
  builds a single, non-augmented, CSV-driven test split. It is a separate module (not a stale
  duplicate) because `eval_model.py` and `test_regression.py` import it directly instead of
  `dataloader.py`.
- `train.py` — classification training loop for the plain VGG16 classifier.
- `train_regression.py` — regression training loop; `--model_to_run` selects between plain
  `VGGReg` (0), `Amygdala` (1), or the `Amygdala_lowroad` dual-pathway model (2).
- `eval_model.py`, `test_regression.py` — load a saved checkpoint and evaluate it
  (classification and regression respectively) on a test split.
- `utils.py` — shared training/evaluation/visualization/checkpoint-loading helpers (confusion
  matrices, precision-recall curves, correlation plots, `save_checkpoint`/`load_checkpoint`,
  a `GaussianBlur` augmentation transform).
- `logger.py` — minimal TensorBoard logger (`Logger`) used by `train.py` and
  `train_regression.py` to log scalars/histograms/images during training. Reimplemented here
  on top of `torch.utils.tensorboard.SummaryWriter` since the original module was not part of
  the source project (see Dependencies).
- `print_model_architectures.ipynb` — notebook that loads saved checkpoints (or a fresh
  `torchvision` VGG16) and prints their architecture/parameter summary via `torchsummary`.
- `requirements.txt` — inferred package list (see Dependencies).

## How to Use

None of these scripts include or download data. IAPS images and their normative valence
ratings are not part of this repository and are not redistributable - supply your own copy,
laid out the way each dataloader expects (below), and pass the matching `--data_dir` /
`--csv_*` command-line arguments (each script's defaults point at the original author's local
paths and will need to be overridden).

### Classification pipeline

1. **`train.py`**
   - Input: `--data_dir` containing `<TRAIN>/<class_name>/*.jpg`, `<VAL>/...`, `<TEST>/...`
     subfolders (an `ImageFolder` layout, one subfolder per class).
   - Does: fine-tunes a VGG16 classifier over `--epoch` epochs, logging scalars/histograms/
     training-image samples via `logger.py` to `./logs`, and evaluating/visualizing
     predictions before and after training.
   - Output: the best checkpoint saved to `--model_dir/--model_name`, plus TensorBoard logs
     under `./logs` (view with `tensorboard --logdir ./logs`).
2. **`eval_model.py`**
   - Input: a checkpoint produced by `train.py` (`--model_dir/--model_name`), and a test image
     folder + selection CSV (`--test_subset`, `--csv_select_subset`).
   - Does: loads the checkpoint and reports classification accuracy/confusion matrix on the
     selected test subset.
   - Output: printed metrics plus matplotlib figures (confusion matrix, precision-recall).

### Regression pipeline

1. **`train_regression.py`**
   - Input: `--data_dir` containing `<TRAIN>/`, `<VAL>/`, `<TEST>/` image folders plus
     matching `--csv_train`/`--csv_val`/`--csv_test` files (image filename + continuous
     valence/arousal-style rating column).
   - Does: trains `VGGReg`, `Amygdala`, or `Amygdala_lowroad` (pick via `--model_to_run`) with
     MSE loss, logging via `logger.py` the same way as `train.py`.
   - Output: the best checkpoint saved to `--model_dir/--model_name`, plus TensorBoard logs
     under the path passed to `Logger(...)` in the script.
2. **`test_regression.py`**
   - Input: a checkpoint from `train_regression.py`, plus a CSV-driven test split
     (`--data_dir`, `--TEST`, `--csv_test`).
   - Does: loads the checkpoint and reports the correlation (Pearson r) and MSE between
     predicted and rated valence/arousal on the test split.
   - Output: printed metrics plus a correlation scatter plot.

### Inspecting a trained model

- **`print_model_architectures.ipynb`** — point `model_dir`/`model_files` at your own saved
  checkpoint(s) (from either training script) and run all cells to print each model's
  layer-by-layer architecture and parameter counts via `torchsummary`. It also works with a
  fresh, untrained `torchvision` VGG16 for comparison (no checkpoint needed for that cell).

## Dependencies

No `requirements.txt` was present in the source project; `requirements.txt` here was written
based on the imports across these files:

```
torch
torchvision
torchsummary
scikit-image
scikit-learn
scipy
seaborn
pandas
numpy
matplotlib
Pillow
tensorboard    # backs logger.py's Logger (torch.utils.tensorboard.SummaryWriter)
torchsampler   # ImbalancedDatasetSampler, used by dataloader.py's cls_dataloader
```

Install with:

```
pip install -r requirements.txt
```

### Note on `logger.py`

`train.py` and `train_regression.py` both do `from logger import Logger`, but no `logger.py`
existed anywhere in the original project - it has been added here as a minimal TensorBoard
logger providing the same `Logger(log_dir)` / `scalar_summary` / `histo_summary` /
`image_summary` interface both scripts already call, so they run end-to-end. View the results
with `tensorboard --logdir <log_dir>`.
