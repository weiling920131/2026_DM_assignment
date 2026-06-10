import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

TRAIN_ROOT = './train/train'
TEST_ROOT  = './test/test'


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
        # ── group: basic ──
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
        # ── group: jerk ──
        jerk = np.diff(v)
        feats[f'{col}_jerk_mean'] = np.mean(np.abs(jerk))
        feats[f'{col}_jerk_std']  = np.std(jerk)
        feats[f'{col}_jerk_max']  = np.max(np.abs(jerk))
        # ── group: fft ──
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
        # ── group: rolling ──
        s = pd.Series(v)
        roll = s.rolling(30, min_periods=1)
        feats[f'{col}_roll_std_mean'] = roll.std().mean()
        feats[f'{col}_roll_std_max']  = roll.std().max()
        feats[f'{col}_roll_mean_std'] = roll.mean().std()
        # ── group: segment ──
        feats.update(segment_features(v, col, n_segs=5))

    # Magnitude (basic)
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


def process_file(fpath, user, is_train):
    df = pd.read_csv(fpath)
    feats = extract_features(df)
    feats['file_id'] = int(df['file_id'].iloc[0])
    feats['user'] = user
    if is_train:
        feats['label'] = int(df['label'].iloc[0])
    return feats


def load_dataset(root, is_train=True):
    jobs = []
    for user in sorted(os.listdir(root)):
        user_path = os.path.join(root, user)
        if not os.path.isdir(user_path):
            continue
        for fname in sorted(os.listdir(user_path)):
            if fname.endswith('.csv'):
                jobs.append((os.path.join(user_path, fname), user))
    print(f"  Found {len(jobs)} files, loading in parallel...")
    records = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_file)(p, u, is_train) for p, u in jobs
    )
    return pd.DataFrame(records)


print("Loading training data (parallel)...")
train_df = load_dataset(TRAIN_ROOT, is_train=True)
print(f"  Train shape: {train_df.shape}")

print("\nLoading test data (parallel)...")
test_df = load_dataset(TEST_ROOT, is_train=False)
print(f"  Test shape:  {test_df.shape}")

train_df.to_pickle('train_features.pkl')
test_df.to_pickle('test_features.pkl')
print("\nSaved train_features.pkl and test_features.pkl")
