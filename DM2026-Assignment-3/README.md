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


## 4. Reproduce the Ablation Study (used in the report)


```bash
# Step 1: extract the full feature set (431 features/file) and cache to disk
python build_features.py        # -> train_features.pkl, test_features.pkl

# Step 2: feature-group / model / ensemble ablation (GroupKFold by user)
python cv_select.py

# Step 3: add subject-wise normalized features (user mean / dev / z-score)
python add_user_features.py     # -> train_features_user.pkl, test_features_user.pkl

# Step 4: ablation over normalization variants (GroupKFold by user)
python cv_select_user.py
```


