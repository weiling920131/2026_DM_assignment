import warnings
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

SEED = 42

train_df = pd.read_pickle('train_features.pkl')
print(f"Train shape: {train_df.shape}")
print(f"Users: {train_df['user'].nunique()}")
print(f"Label distribution:\n{train_df['label'].value_counts().sort_index()}\n")

all_cols = [c for c in train_df.columns if c not in ('file_id', 'user', 'label')]


def in_group(col, keys):
    return any(k in col for k in keys)


ROLLING = [c for c in all_cols if 'roll' in c]
SEGMENT = [c for c in all_cols if 'seg' in c]
JERK    = [c for c in all_cols if 'jerk' in c]
FFT     = [c for c in all_cols if in_group(c, ['fft', 'dom_freq', 'dom_amp', 'spec_entropy'])]
BASIC   = [c for c in all_cols if c not in ROLLING + SEGMENT + JERK + FFT]

print(f"BASIC:   {len(BASIC)} features")
print(f"FFT:     {len(FFT)} features")
print(f"JERK:    {len(JERK)} features")
print(f"ROLLING: {len(ROLLING)} features")
print(f"SEGMENT: {len(SEGMENT)} features\n")

y = train_df['label'].values
groups = train_df['user'].values
gkf = GroupKFold(n_splits=5)


def cv_f1(feature_cols, model_fn, name):
    X = np.nan_to_num(train_df[feature_cols].values.astype(np.float32))
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = model_fn()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        f1 = f1_score(y_va, pred, average='macro')
        scores.append(f1)
    print(f"{name:40s} | feats={len(feature_cols):4d} | "
          f"F1 macro = {np.mean(scores):.4f} +/- {np.std(scores):.4f} | folds={[f'{s:.3f}' for s in scores]}")
    return np.mean(scores)


def quick_lgbm():
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05,
        num_leaves=63, max_depth=-1,
        subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, min_child_samples=5,
        class_weight='balanced',
        n_jobs=-1, random_state=SEED, verbose=-1
    )


print("=" * 70)
print("STEP 1: Feature group comparison (GroupKFold by user, quick LGBM)")
print("=" * 70)

feature_sets = {
    'BASIC':                       BASIC,
    'BASIC+FFT':                   BASIC + FFT,
    'BASIC+FFT+JERK':              BASIC + FFT + JERK,
    'BASIC+FFT+JERK+ROLLING':      BASIC + FFT + JERK + ROLLING,
    'BASIC+FFT+JERK+SEGMENT':      BASIC + FFT + JERK + SEGMENT,
    'ALL (BASIC+FFT+JERK+ROLL+SEG)': all_cols,
}

results = {}
for name, cols in feature_sets.items():
    results[name] = (cv_f1(cols, quick_lgbm, name), cols)

best_name = max(results, key=lambda k: results[k][0])
best_cols = results[best_name][1]
print(f"\nBest feature set: {best_name} (F1={results[best_name][0]:.4f})\n")


print("=" * 70)
print("STEP 2: Model comparison on best feature set")
print("=" * 70)


def lgbm1():
    return lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.05,
        num_leaves=63, max_depth=-1,
        subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, min_child_samples=5,
        class_weight='balanced', reg_alpha=0.1, reg_lambda=0.1,
        n_jobs=-1, random_state=SEED, verbose=-1
    )

def lgbm2():
    return lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.03,
        num_leaves=127, max_depth=-1,
        subsample=0.7, subsample_freq=1,
        colsample_bytree=0.7, min_child_samples=5,
        class_weight='balanced', reg_alpha=0.05, reg_lambda=0.05,
        n_jobs=-1, random_state=SEED + 1, verbose=-1
    )

def rf():
    return RandomForestClassifier(
        n_estimators=400, max_depth=None,
        min_samples_leaf=2, max_features='sqrt',
        class_weight='balanced_subsample',
        n_jobs=-1, random_state=SEED
    )

def xgbm():
    return xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.05,
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss',
        n_jobs=-1, random_state=SEED, verbosity=0
    )


cv_f1(best_cols, lgbm1, 'LightGBM-1')
cv_f1(best_cols, lgbm2, 'LightGBM-2')
cv_f1(best_cols, rf,    'RandomForest')
cv_f1(best_cols, xgbm,  'XGBoost')


print("\n" + "=" * 70)
print("STEP 3: Ensemble (soft voting) on best feature set")
print("=" * 70)

X = np.nan_to_num(train_df[best_cols].values.astype(np.float32))


def cv_ensemble(weights, name):
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        m1, m2, m3, m4 = lgbm1(), lgbm2(), rf(), xgbm()
        m1.fit(X_tr, y_tr)
        m2.fit(X_tr, y_tr)
        m3.fit(X_tr, y_tr)
        m4.fit(X_tr, y_tr)

        p1 = m1.predict_proba(X_va)
        p2 = m2.predict_proba(X_va)
        p3 = m3.predict_proba(X_va)
        p4 = m4.predict_proba(X_va)

        avg = weights[0]*p1 + weights[1]*p2 + weights[2]*p3 + weights[3]*p4
        pred = np.argmax(avg, axis=1)
        f1 = f1_score(y_va, pred, average='macro')
        scores.append(f1)
    print(f"{name:40s} | weights={weights} | "
          f"F1 macro = {np.mean(scores):.4f} +/- {np.std(scores):.4f} | folds={[f'{s:.3f}' for s in scores]}")
    return np.mean(scores)


cv_ensemble([0.35, 0.35, 0.15, 0.15], 'Ensemble (orig weights)')
cv_ensemble([0.5, 0.5, 0.0, 0.0],     'Ensemble (LGBM only, equal)')
cv_ensemble([0.25, 0.25, 0.25, 0.25], 'Ensemble (equal all)')

print("\nDone. Use the best feature set + model/ensemble combo for final submission.")
