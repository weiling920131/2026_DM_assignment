import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

SEED = 42
SAMPLE_SUB = './sample_submission.csv'
OUTPUT     = './submission_final.csv'

train_df = pd.read_pickle('train_features_user.pkl')
test_df  = pd.read_pickle('test_features_user.pkl')
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

orig_cols = [c for c in train_df.columns
              if c not in ('file_id', 'user', 'label')
              and not c.endswith(('_userdev', '_userz', '_usermean'))]
z_cols    = [c for c in train_df.columns if c.endswith('_userz')]
feature_cols = orig_cols + z_cols
print(f"Final feature count: {len(feature_cols)} (ORIG={len(orig_cols)} + USERZ={len(z_cols)})")

X      = np.nan_to_num(train_df[feature_cols].values.astype(np.float32))
y      = train_df['label'].values
groups = train_df['user'].values
X_test = np.nan_to_num(test_df[feature_cols].values.astype(np.float32))


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

def xgbm():
    return xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.05,
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss',
        n_jobs=-1, random_state=SEED, verbosity=0
    )


WEIGHTS = [0.4, 0.4, 0.2]  # lgbm1, lgbm2, xgb

print("\n" + "=" * 60)
print("Final CV check (GroupKFold by user, full hyperparams)")
print("=" * 60)

gkf = GroupKFold(n_splits=5)
scores = []
for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    m1, m2, m3 = lgbm1(), lgbm2(), xgbm()
    m1.fit(X_tr, y_tr)
    m2.fit(X_tr, y_tr)
    m3.fit(X_tr, y_tr)

    p1 = m1.predict_proba(X_va)
    p2 = m2.predict_proba(X_va)
    p3 = m3.predict_proba(X_va)

    avg = WEIGHTS[0]*p1 + WEIGHTS[1]*p2 + WEIGHTS[2]*p3
    pred = np.argmax(avg, axis=1)
    f1 = f1_score(y_va, pred, average='macro')
    scores.append(f1)
    print(f"  Fold {fold}: F1 macro = {f1:.4f}")

print(f"\nFinal CV F1 macro = {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

print("\n" + "=" * 60)
print("Training on full training set...")
print("=" * 60)

m1, m2, m3 = lgbm1(), lgbm2(), xgbm()
print("  LightGBM-1...")
m1.fit(X, y)
print("  LightGBM-2...")
m2.fit(X, y)
print("  XGBoost...")
m3.fit(X, y)

p1 = m1.predict_proba(X_test)
p2 = m2.predict_proba(X_test)
p3 = m3.predict_proba(X_test)

avg_prob = WEIGHTS[0]*p1 + WEIGHTS[1]*p2 + WEIGHTS[2]*p3
preds = np.argmax(avg_prob, axis=1)

sub = pd.read_csv(SAMPLE_SUB)
pred_map = dict(zip(test_df['file_id'].values, preds))
sub['Label'] = sub['Id'].map(pred_map)
missing = sub['Label'].isna().sum()
if missing > 0:
    print(f"WARNING: {missing} missing — filling with 0")
    sub['Label'] = sub['Label'].fillna(0).astype(int)
else:
    sub['Label'] = sub['Label'].astype(int)

sub.to_csv(OUTPUT, index=False)
print(f"\nSaved: {OUTPUT}")
print(f"Prediction distribution:\n{sub['Label'].value_counts().sort_index()}")
print("\nDone!")
