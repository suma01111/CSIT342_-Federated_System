# Video 
phase 1 : https://drive.google.com/file/d/1-7CJTI5-nSLhDFe36UUTiBZNfHore6P-/view?usp=drive_link
phase 2: https://drive.google.com/file/d/1QstSIEdoAT7uoLPJKO6imcQHaM1_5WWh/view?usp=sharing

# Federated Breast Cancer Classification

This project demonstrates a federated learning approach for breast cancer image classification using a Vision Transformer (ViT) model and PyTorch.

## Project Overview

- `train.py`: Trains a local breast cancer classifier on a binary dataset of benign vs malign images and saves the model as `breast_cancer_model.pth`.
- `federated_train.py`: Simulates federated learning with multiple clients, secure aggregation, differential privacy, and shared training of a global model.
- `predict.py`: Loads a trained model and predicts whether a sample image is benign or malignant.
- `compare_models.py`: Compare the performance of different trained models.
- `compare_prediction.py`: Compare predictions from different models or inference runs.
- `plot_results.py`: Plot training or evaluation results.
- `secure_plot_results.py`: Plot results for secure/federated training runs.

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

> This project uses PyTorch, torchvision, timm, scikit-learn, and related packages.

## Dataset

The code expects the dataset to be organized under:

```text
dataset/bus_uclm_separated/
```

with subfolders for image classes such as `benign` and `malign`.

## Usage

### Train a local model

```bash
python train.py
```

This will train a ViT-based image classifier and save the model weights to `breast_cancer_model.pth`.

### Run federated training

```bash
python federated_train.py
```

This script simulates federated training among multiple clients and writes results to the `results/` directory.

### Predict on a sample image

Update `MODEL_PATH` and `IMAGE_PATH` inside `predict.py`, then run:

```bash
python predict.py
```

### Compare models and results

- `compare_models.py`: Compare saved models or model outputs.
- `compare_prediction.py`: Compare predictions across different inputs.
- `plot_results.py` / `secure_plot_results.py`: Generate result visualizations.

## Outputs

The repository includes a `results/` directory containing example output files such as:

- `federated_results_summary.json`
- `federated_results.csv`
- `federated_results.html`

## Notes

- The training scripts use GPU if available, otherwise fall back to CPU.
- The project filters the dataset to binary labels `benign` and `malign`.
- `federated_train.py` includes configurable privacy settings like clipping, local DP noise, and secure aggregation.


