# NLP Coursework: Detecting Patronising and Condescending Language (PCL)

---

## Overview

This project fine-tunes `roberta-large` to detect Patronising and Condescending Language (PCL) in news articles targeting vulnerable communities. The model achieves an **F1 score of 0.63** on the official dev set, compared to the RoBERTa-base baseline of 0.48.

### Key Contributions
- Fine-tuned `roberta-large` (vs baseline `roberta-base`)
- Focal Loss to handle severe class imbalance (~10% positive)
- Keyword-prefix strategy: prepends the topic keyword (e.g. `Topic: homeless.`) to each input text
- Threshold optimisation: threshold sweep at inference time to maximise F1 rather than using a fixed 0.5 cutoff
- Cosine learning rate scheduler with warmup for stable large model fine-tuning

## Model Weights

The trained model checkpoint is too large to store in Git.

**Download the full checkpoint folder from:** *https://drive.google.com/drive/folders/1ciEL_Avr_dhefk9UxXj-GBQBfTCHy0gN?usp=share_link*

After downloading, place the entire folder at:
```
Best_Model/best_model_checkpoint/
```
---

## Results

| Model | Dev F1 | Test F1 |
|---|---|---|
| RoBERTa-base (baseline) | 0.48 | 0.49 |
| **Ours (RoBERTa-large + Focal Loss + Keyword Prefix)** | **0.63** | **TBD** |

### Ablation Study

| Configuration | Dev F1 | Change |
|---|---|---|
| Full model (keyword prefix + max_length=256) | 0.6265 | — |
| Without keyword prefix | 0.6005 | −0.0259 |
| max_length=128 | 0.6285 | +0.0020 |

---

## Repository Structure

```
├── Best_Model/
│   └── model_training.ipynb               # Full training, evaluation and inference notebook
├── plots/                        # Figures generated during EDA and evaluation
│   ├── confusion_matrix.png
│   ├── class_distribution.png
│   ├── ngram_plot.png
│   ├── sequence_length_distribution.png
│   ├── error_by_keyword.png
│   └── pr_curve.png
├── predictions/
│   ├── dev.txt                   # Dev set predictions (one 0/1 per line)
│   └── test.txt                  # Test set predictions (one 0/1 per line)
├── eda-analysis.ipynb            # Exploratory Data Analysis notebook
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

---

## Data

The dataset is the **Don't Patronize Me!** corpus from SemEval 2022 Task 4.

| File | Link |
|---|---|
| Full dataset (`dontpatronizeme_pcl.tsv`) | [Task Repository](https://github.com/Perez-AlmendrosC/dontpatronizeme) |
| Train split IDs (`train_semeval_parids-labels.csv`) | [Task Repository](https://github.com/Perez-AlmendrosC/dontpatronizeme) |
| Dev split IDs (`dev_semeval_parids-labels.csv`) | [Task Repository](https://github.com/Perez-AlmendrosC/dontpatronizeme) |
| Test set without labels (`task4_test.tsv`) | [Task Repository](https://github.com/Perez-AlmendrosC/dontpatronizeme) |

Download all files and place them in a `data/` directory at the project root. Update `DATA_PATH` in the notebook accordingly.

**Label convention:** Labels >= 2 are treated as PCL (positive class = 1). Labels 0-1 are treated as No PCL (negative class = 0).

---

## Environment Setup

### Requirements

- Python 3.12
- CUDA-capable GPU (trained on Azure with A100)
- ~20GB GPU memory for `roberta-large` with `max_length=256`

### Installation

```bash
# 1. Create virtual environment
python -m venv nlp_venv
source nlp_venv/bin/activate  # Linux/Mac

# 2. Install PyTorch with CUDA support (must be done separately)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install all other dependencies
pip install -r requirements.txt
```

> **Note:** PyTorch must be installed before `requirements.txt` because the correct CUDA version depends on your system. The `requirements.txt` file covers all other dependencies.

---

## Reproducing Results

### Training

Open and run `Best_Model/train.ipynb` in order. The notebook covers:
1. Data loading and preprocessing
2. Tokenisation with keyword prefix
3. Model training with Focal Loss
4. Evaluation and threshold selection
5. Generating `dev.txt` and `test.txt`

Update `DATA_PATH` at the top of the notebook to point to your local data directory.

### Key Hyperparameters

| Parameter | Value |
|---|---|
| Model | roberta-large |
| Epochs | 6 (early stopping patience=2) |
| Learning rate | 8e-6 |
| Batch size | 8 (effective: 32 with gradient accumulation steps=4) |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| LR scheduler | cosine |
| Loss function | Focal Loss (alpha=1-pos_ratio, gamma=2) |
| fp16 | True |
| max_length | 256 |

### Inference Only (skip retraining)

If you only want to regenerate predictions using the saved checkpoint:

```python
# 1. Download model.safetensors and place in Best_Model/best_model_checkpoint/
# 2. Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Best_Model/best_model_checkpoint", local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained("Best_Model/best_model_checkpoint", local_files_only=True)
model = model.to("cuda")

# 3. Run inference - see train.ipynb for full generate_predictions() function
# Optimal threshold for this model is 0.85
dev_preds  = generate_predictions(model, encoded_dev,  threshold=0.85)
test_preds = generate_predictions(model, encoded_test, threshold=0.85)
```

---

## Prediction File Format

Both `predictions/dev.txt` and `predictions/test.txt` contain one integer per line:
- `0` = No PCL
- `1` = PCL


---

## EDA

Exploratory data analysis is in `eda-analysis.ipynb`. It covers:
- Class distribution and imbalance analysis
- Token length distribution
- Keyword frequency analysis
- Noise identification (HTML entities, URLs, duplicates)

---

## Error Analysis and Local Evaluation

Error analysis and all local evaluation code is in `Best_Model/model_training.ipynb`, in the sections following model training. It covers:

- **Confusion matrix** on the dev set
- **Precision-Recall curve** with the chosen classification threshold marked
- **False positive and false negative inspection** — concrete examples with text, keyword, and predicted probability
- **Error distribution by keyword** — bar chart showing which topic keywords (e.g. homeless, refugee) the model struggles with most, broken down by false positives and false negatives
- **Ablation study** — effect of removing the keyword prefix and reducing max_length from 256 to 128

All generated plots are saved in the `plots/` directory.

## Notes on Reproducibility

- Seed is fixed at 42 across `torch`, `numpy`, `random`, and `transformers` for reproducibility
- Minor F1 variation (~+-0.005) may still occur across different hardware or CUDA versions
- Results were obtained on an Azure VM with a single A100 GPU
- The threshold sweep in `compute_metrics` selects the best threshold per epoch on the dev set; the optimal threshold found was 0.85