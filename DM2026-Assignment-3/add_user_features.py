import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

train_df = pd.read_pickle('train_features.pkl')
test_df  = pd.read_pickle('test_features.pkl')

base_cols = [c for c in train_df.columns if c not in ('file_id', 'user', 'label')]

print(f"Base feature count: {len(base_cols)}")
print("Adding per-user (subject-wise) normalized features...")


def add_user_features(df, base_cols):
    df = df.copy()
    g = df.groupby('user')[base_cols]
    user_mean = g.transform('mean')
    user_std  = g.transform('std').replace(0, 1).fillna(1)

    dev_df    = (df[base_cols] - user_mean)
    dev_df.columns    = [f'{c}_userdev'  for c in base_cols]
    zscore_df = (df[base_cols] - user_mean) / user_std
    zscore_df.columns = [f'{c}_userz'    for c in base_cols]

    user_mean.columns = [f'{c}_usermean' for c in base_cols]

    return pd.concat([df, user_mean, dev_df, zscore_df], axis=1)


train_aug = add_user_features(train_df, base_cols)
test_aug  = add_user_features(test_df, base_cols)

print(f"Train shape: {train_df.shape} -> {train_aug.shape}")
print(f"Test shape:  {test_df.shape} -> {test_aug.shape}")

train_aug.to_pickle('train_features_user.pkl')
test_aug.to_pickle('test_features_user.pkl')
print("\nSaved train_features_user.pkl and test_features_user.pkl")
