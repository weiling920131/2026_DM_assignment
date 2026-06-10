# 2026_DM_assignment-3

NYCU Data Mining Spring 2026 — Assignment 3: Human Activity Recognition (Kaggle Competition)

## 1. Environment Setup

```bash
conda create -n dmas3 python=3.10 -y
conda activate dmas3
pip install -r requirements.txt
```

## 2. Data Setup

Download the competition data from Kaggle
```
DM2026-Assignment-3/
├── train/train/User_001/...csv ... User_060/...csv
├── test/test/User_061/...csv  ... User_100/...csv
├── sample_submission.csv
```

## 3. Reproduce the Best Submission (public score 0.7719)

```bash
python train_predict_fast.py
```




