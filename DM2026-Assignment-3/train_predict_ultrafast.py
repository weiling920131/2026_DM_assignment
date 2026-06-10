import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
from joblib import Parallel, delayed
import lightgbm as lgb

warnings.filterwarnings('ignore')

TRAIN_ROOT = './train/train'
TEST_ROOT  = './test/test'
SAMPLE_SUB = './sample_submission.csv'
OUTPUT     = './submission_fast.csv'
SEED       = 42


def extract_features(df):
    feats = {}
    axes = ['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z']

    for col in axes:
        v = df[col].values.astype(np.float64)
        feats[f'{col}_mean']   = np.mean(v)
        feats[f'{col}_std']    = np.std(v)
        feats[f'{col}_min']    = np.min(v)
        feats[f'{col}_max']    = np.max(v)
        feats[f'{col}_range']  = np.max(v) - np.min(v)
        feats[f'{col}_median'] = np.median(v)
        feats[f'{col}_p25']    = np.percentile(v, 25)
        feats[f'{col}_p75']    = np.percentile(v, 75)
        feats[f'{col}_iqr']    = np.percentile(v, 75) - np.percentile(v, 25)
        feats[f'{col}_skew']   = stats.skew(v)
        feats[f'{col}_kurt']   = stats.kurtosis(v)
        feats[f'{col}_rms']    = np.sqrt(np.mean(v**2))
        feats[f'{col}_mad']    = np.mean(np.abs(v - np.mean(v)))
        v_dm = v - np.mean(v)
        feats[f'{col}_zcr']    = int(np.sum((v_dm[:-1] * v_dm[1:]) < 0))
        if np.std(v) > 1e-8:
            feats[f'{col}_ac1'] = np.corrcoef(v[:-1], v[1:])[0, 1]
        else:
            feats[f'{col}_ac1'] = 0.0
        fft_mag = np.abs(np.fft.rfft(v))[1:]
        feats[f'{col}_fft_energy']   = np.sum(fft_mag**2)
        feats[f'{col}_fft_mean']     = np.mean(fft_mag)
        feats[f'{col}_fft_std']      = np.std(fft_mag)
        feats[f'{col}_fft_max']      = np.max(fft_mag)
        top3_idx = np.argsort(fft_mag)[-3:][::-1]
        for rank, idx in enumerate(top3_idx):
            feats[f'{col}_dom_freq_{rank}'] = idx
            feats[f'{col}_dom_amp_{rank}']  = fft_mag[idx]
        psd = fft_mag**2
        psd_norm = psd / (np.sum(psd) + 1e-9)
        feats[f'{col}_spec_entropy'] = -np.sum(psd_norm * np.log(psd_norm + 1e-9))

    mag = np.sqrt(df['mean_x']**2 + df['mean_y']**2 + df['mean_z']**2).values
    feats['mag_mean']       = np.mean(mag)
    feats['mag_std']        = np.std(mag)
    feats['mag_range']      = np.max(mag) - np.min(mag)
    feats['mag_max']        = np.max(mag)
    feats['mag_median']     = np.median(mag)
    feats['mag_iqr']        = np.percentile(mag, 75) - np.percentile(mag, 25)
    feats['mag_skew']       = stats.skew(mag)
    feats['mag_kurt']       = stats.kurtosis(mag)
    feats['mag_rms']        = np.sqrt(np.mean(mag**2))
    fft_m = np.abs(np.fft.rfft(mag))[1:]
    feats['mag_fft_energy'] = np.sum(fft_m**2)
    feats['mag_fft_mean']   = np.mean(fft_m)
    feats['mag_fft_max']    = np.max(fft_m)
    v_dm = mag - np.mean(mag)
    feats['mag_zcr']        = int(np.sum((v_dm[:-1] * v_dm[1:]) < 0))
    feats['sma'] = np.mean(
        np.abs(df['mean_x']) + np.abs(df['mean_y']) + np.abs(df['mean_z'])
    )

    for (a, b) in [('mean_x', 'mean_y'), ('mean_x', 'mean_z'), ('mean_y', 'mean_z')]:
        va, vb = df[a].values, df[b].values
        if np.std(va) > 1e-8 and np.std(vb) > 1e-8:
            feats[f'corr_{a}_{b}'] = np.corrcoef(va, vb)[0, 1]
        else:
            feats[f'corr_{a}_{b}'] = 0.0

    ax, ay, az = np.mean(df['mean_x']), np.mean(df['mean_y']), np.mean(df['mean_z'])
    norm = np.sqrt(ax**2 + ay**2 + az**2) + 1e-9
    feats['tilt_x'] = ax / norm
    feats['tilt_y'] = ay / norm
    feats['tilt_z'] = az / norm

    for col in ['mean_x', 'mean_y', 'mean_z']:
        jerk = np.diff(df[col].values)
        feats[f'{col}_jerk_mean'] = np.mean(np.abs(jerk))
        feats[f'{col}_jerk_std']  = np.std(jerk)
        feats[f'{col}_jerk_max']  = np.max(np.abs(jerk))

    return feats


def process_file(fpath, is_train):
    df = pd.read_csv(fpath)
    feats = extract_features(df)
    feats['file_id'] = int(df['file_id'].iloc[0])
    if is_train:
        feats['label'] = int(df['label'].iloc[0])
    return feats


def load_dataset(root, is_train=True):
    all_paths = []
    for user in sorted(os.listdir(root)):
        user_path = os.path.join(root, user)
        if not os.path.isdir(user_path):
            continue
        for fname in sorted(os.listdir(user_path)):
            if fname.endswith('.csv'):
                all_paths.append(os.path.join(user_path, fname))

    print(f"  Found {len(all_paths)} files, loading in parallel...")
    records = Parallel(n_jobs=-1, verbose=0)(
        delayed(process_file)(p, is_train) for p in all_paths
    )
    return pd.DataFrame(records)


print("Loading training data (parallel)...")
train_df = load_dataset(TRAIN_ROOT, is_train=True)
print(f"  Train shape: {train_df.shape}")
print(f"  Label distribution:\n{train_df['label'].value_counts().sort_index()}")

print("\nLoading test data (parallel)...")
test_df = load_dataset(TEST_ROOT, is_train=False)
print(f"  Test shape:  {test_df.shape}")

feature_cols = [c for c in train_df.columns if c not in ('file_id', 'label')]
X      = np.nan_to_num(train_df[feature_cols].values.astype(np.float32))
y      = train_df['label'].values
X_test = np.nan_to_num(test_df[feature_cols].values.astype(np.float32))
print(f"\nFeature count: {X.shape[1]}")

model = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05,
    num_leaves=63, max_depth=-1,
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, min_child_samples=5,
    class_weight='balanced', reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, random_state=SEED, verbose=-1
)

print("\nTraining LightGBM...")
model.fit(X, y)
print("  Done.")

preds = model.predict(X_test)

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
