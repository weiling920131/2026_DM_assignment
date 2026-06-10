import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

SEED = 42

train_df = pd.read_pickle('train_features_user.pkl')
print(f"Train shape: {train_df.shape}")

base_cols    = [c for c in train_df.columns if c.endswith(('_userdev', '_userz', '_usermean'))]
orig_cols    = [c for c in train_df.columns
                if c not in ('file_id', 'user', 'label') and c not in base_cols]
dev_cols     = [c for c in train_df.columns if c.endswith('_userdev')]
z_cols       = [c for c in train_df.columns if c.endswith('_userz')]
mean_cols    = [c for c in train_df.columns if c.endswith('_usermean')]

print(f"ORIG: {len(orig_cols)}, DEV: {len(dev_cols)}, Z: {len(z_cols)}, USERMEAN: {len(mean_cols)}")

y = train_df['label'].values
groups = train_df['user'].values
gkf = GroupKFold(n_splits=5)


def quick_lgbm():
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05,
        num_leaves=63, max_depth=-1,
        subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, min_child_samples=5,
        class_weight='balanced',
        n_jobs=-1, random_state=SEED, verbose=-1
    )


def cv_f1(feature_cols, name):
    X = np.nan_to_num(train_df[feature_cols].values.astype(np.float32))
    scores = []
    for tr_idx, va_idx in gkf.split(X, y, groups):
        model = quick_lgbm()
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[va_idx])
        scores.append(f1_score(y[va_idx], pred, average='macro'))
    print(f"{name:35s} | feats={len(feature_cols):4d} | "
          f"F1 macro = {np.mean(scores):.4f} +/- {np.std(scores):.4f} | folds={[f'{s:.3f}' for s in scores]}")
    return np.mean(scores)


print("=" * 70)
print("Comparing feature sets WITH/WITHOUT subject-wise normalization")
print("=" * 70)

cv_f1(orig_cols,                                  'ORIG only (baseline)')
cv_f1(orig_cols + dev_cols,                       'ORIG + USERDEV')
cv_f1(orig_cols + z_cols,                         'ORIG + USERZ')
cv_f1(orig_cols + mean_cols,                      'ORIG + USERMEAN')
cv_f1(orig_cols + dev_cols + mean_cols,           'ORIG + USERDEV + USERMEAN')
cv_f1(orig_cols + z_cols + mean_cols,             'ORIG + USERZ + USERMEAN')
cv_f1(dev_cols,                                   'USERDEV only')
cv_f1(z_cols,                                     'USERZ only')

print("\nCompare these to STEP1 results from cv_select.py to see if user-wise")
print("normalization improves cross-user generalization.")
