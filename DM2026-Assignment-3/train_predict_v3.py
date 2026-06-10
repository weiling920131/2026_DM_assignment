import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

TRAIN_ROOT = './train/train'
TEST_ROOT  = './test/test'
SAMPLE_SUB = './sample_submission.csv'
OUTPUT     = './submission_v3.csv'
SEED       = 42


def segment_features(v, col, n_segs=5):
    feats = {}
    seg_len = len(v) // n_segs
    seg_means, seg_stds = [], []
    for i in range(n_segs):
        seg = v[i * seg_len: (i + 1) * seg_len]
        seg_means.append(np.mean(seg))
        seg_stds.append(np.std(seg))
        feats[f'{col}_seg{i}_mean'] = np.mean(seg)
        feats[f'{col}_seg{i}_std']  = np.std(seg)
        feats[f'{col}_seg{i}_max']  = np.max(seg)
        feats[f'{col}_seg{i}_min']  = np.min(seg)
    feats[f'{col}_seg_trend']     = seg_means[-1] - seg_means[0]
    feats[f'{col}_seg_std_trend'] = seg_stds[-1] - seg_stds[0]
    feats[f'{col}_seg_mean_std']  = np.std(seg_means)
    feats[f'{col}_seg_std_std']   = np.std(seg_stds)
    return feats


def extract_features(df):
    feats = {}
    axes = ['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z']
    raw = {}

    for col in axes:
        v = df[col].values.astype(np.float64)
        raw[col] = v
        feats[f'{col}_mean']   = np.mean(v)
        feats[f'{col}_std']    = np.std(v)
        feats[f'{col}_min']    = np.min(v)
        feats[f'{col}_max']    = np.max(v)
        feats[f'{col}_range']  = np.max(v) - np.min(v)
        feats[f'{col}_median'] = np.median(v)
        feats[f'{col}_p5']     = np.percentile(v, 5)
        feats[f'{col}_p10']    = np.percentile(v, 10)
        feats[f'{col}_p25']    = np.percentile(v, 25)
        feats[f'{col}_p75']    = np.percentile(v, 75)
        feats[f'{col}_p90']    = np.percentile(v, 90)
        feats[f'{col}_p95']    = np.percentile(v, 95)
        feats[f'{col}_iqr']    = np.percentile(v, 75) - np.percentile(v, 25)
        feats[f'{col}_skew']   = stats.skew(v)
        feats[f'{col}_kurt']   = stats.kurtosis(v)
        feats[f'{col}_rms']    = np.sqrt(np.mean(v**2))
        feats[f'{col}_mad']    = np.mean(np.abs(v - np.mean(v)))
        m = np.mean(v)
        feats[f'{col}_cv']     = np.std(v) / (abs(m) + 1e-9)
        v_dm = v - np.mean(v)
        feats[f'{col}_zcr']    = int(np.sum((v_dm[:-1] * v_dm[1:]) < 0))
        for lag in [1, 5, 10]:
            if np.std(v) > 1e-8 and len(v) > lag:
                feats[f'{col}_ac{lag}'] = np.corrcoef(v[:-lag], v[lag:])[0, 1]
            else:
                feats[f'{col}_ac{lag}'] = 0.0
        # FFT
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
        # Rolling stats (window=30s)
        s = pd.Series(v)
        roll = s.rolling(30, min_periods=1)
        feats[f'{col}_roll_std_mean'] = roll.std().mean()
        feats[f'{col}_roll_std_max']  = roll.std().max()
        feats[f'{col}_roll_mean_std'] = roll.mean().std()
        # Jerk
        jerk = np.diff(v)
        feats[f'{col}_jerk_mean'] = np.mean(np.abs(jerk))
        feats[f'{col}_jerk_std']  = np.std(jerk)
        feats[f'{col}_jerk_max']  = np.max(np.abs(jerk))
        # 5-segment temporal features
        feats.update(segment_features(v, col, n_segs=5))

    # Magnitude
    mag = np.sqrt(df['mean_x']**2 + df['mean_y']**2 + df['mean_z']**2).values
    feats['mag_mean']       = np.mean(mag)
    feats['mag_std']        = np.std(mag)
    feats['mag_range']      = np.max(mag) - np.min(mag)
    feats['mag_max']        = np.max(mag)
    feats['mag_median']     = np.median(mag)
    feats['mag_p5']         = np.percentile(mag, 5)
    feats['mag_iqr']        = np.percentile(mag, 75) - np.percentile(mag, 25)
    feats['mag_p95']        = np.percentile(mag, 95)
    feats['mag_skew']       = stats.skew(mag)
    feats['mag_kurt']       = stats.kurtosis(mag)
    feats['mag_rms']        = np.sqrt(np.mean(mag**2))
    feats['mag_cv']         = np.std(mag) / (abs(np.mean(mag)) + 1e-9)
    fft_m = np.abs(np.fft.rfft(mag))[1:]
    feats['mag_fft_energy'] = np.sum(fft_m**2)
    feats['mag_fft_mean']   = np.mean(fft_m)
    feats['mag_fft_max']    = np.max(fft_m)
    v_dm = mag - np.mean(mag)
    feats['mag_zcr']        = int(np.sum((v_dm[:-1] * v_dm[1:]) < 0))
    feats.update(segment_features(mag, 'mag', n_segs=5))

    # SMA
    feats['sma'] = np.mean(
        np.abs(df['mean_x']) + np.abs(df['mean_y']) + np.abs(df['mean_z'])
    )

    # Cross-axis correlations
    for (a, b) in [('mean_x', 'mean_y'), ('mean_x', 'mean_z'), ('mean_y', 'mean_z')]:
        va, vb = df[a].values, df[b].values
        if np.std(va) > 1e-8 and np.std(vb) > 1e-8:
            feats[f'corr_{a}_{b}'] = np.corrcoef(va, vb)[0, 1]
        else:
            feats[f'corr_{a}_{b}'] = 0.0

    # Tilt
    ax, ay, az = np.mean(df['mean_x']), np.mean(df['mean_y']), np.mean(df['mean_z'])
    norm = np.sqrt(ax**2 + ay**2 + az**2) + 1e-9
    feats['tilt_x'] = ax / norm
    feats['tilt_y'] = ay / norm
    feats['tilt_z'] = az / norm

    # Energy ratios between axes
    e_x, e_y, e_z = np.sum(raw['mean_x']**2), np.sum(raw['mean_y']**2), np.sum(raw['mean_z']**2)
    e_total = e_x + e_y + e_z + 1e-9
    feats['energy_ratio_x'] = e_x / e_total
    feats['energy_ratio_y'] = e_y / e_total
    feats['energy_ratio_z'] = e_z / e_total

    s_x, s_y, s_z = np.sum(raw['std_x']**2), np.sum(raw['std_y']**2), np.sum(raw['std_z']**2)
    s_total = s_x + s_y + s_z + 1e-9
    feats['std_energy_ratio_x'] = s_x / s_total
    feats['std_energy_ratio_y'] = s_y / s_total
    feats['std_energy_ratio_z'] = s_z / s_total

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

class_counts = Counter(y)
n_samples, n_classes = len(y), 6
class_weights = {c: n_samples / (n_classes * cnt) for c, cnt in class_counts.items()}
sample_weights = np.array([class_weights[yi] for yi in y])
cb_class_weights = [class_weights[c] for c in range(n_classes)]

lgbm1 = lgb.LGBMClassifier(
    n_estimators=800, learning_rate=0.05,
    num_leaves=63, max_depth=-1,
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, min_child_samples=5,
    class_weight='balanced', reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, random_state=SEED, verbose=-1
)

lgbm2 = lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.03,
    num_leaves=127, max_depth=-1,
    subsample=0.7, subsample_freq=1,
    colsample_bytree=0.7, min_child_samples=5,
    class_weight='balanced', reg_alpha=0.05, reg_lambda=0.05,
    n_jobs=-1, random_state=SEED + 1, verbose=-1
)

rf = RandomForestClassifier(
    n_estimators=400, max_depth=None,
    min_samples_leaf=2, max_features='sqrt',
    class_weight='balanced_subsample',
    n_jobs=-1, random_state=SEED
)

xgbm = xgb.XGBClassifier(
    n_estimators=600, learning_rate=0.05,
    max_depth=6, subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss',
    n_jobs=-1, random_state=SEED, verbosity=0
)

cb = CatBoostClassifier(
    iterations=600, learning_rate=0.05,
    depth=6, l2_leaf_reg=3,
    class_weights=cb_class_weights,
    random_seed=SEED, verbose=False,
    thread_count=-1
)

print("\nTraining models...")
print("  LightGBM-1...")
lgbm1.fit(X, y)
print("  LightGBM-2...")
lgbm2.fit(X, y)
print("  RandomForest...")
rf.fit(X, y)
print("  XGBoost...")
xgbm.fit(X, y, sample_weight=sample_weights)
print("  CatBoost...")
cb.fit(X, y)
print("  All done.")

prob1 = lgbm1.predict_proba(X_test)
prob2 = lgbm2.predict_proba(X_test)
prob3 = rf.predict_proba(X_test)
prob4 = xgbm.predict_proba(X_test)
prob5 = cb.predict_proba(X_test)

avg_prob = 0.25 * prob1 + 0.25 * prob2 + 0.10 * prob3 + 0.20 * prob4 + 0.20 * prob5
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
