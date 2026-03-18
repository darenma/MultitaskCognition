import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import nibabel as nib
from volumentations import *

# =====================
# Dataset
# =====================

class CogDataset3d(Dataset):
    def __init__(self, input_dir, target_dir, files, df, transform=False):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.files = sorted(files)
        self.df = df
        self.transform = transform

        # Keep only samples with labels (IMPORTANT)
        valid_files = set(df['filenames'].values)
        self.files = [f for f in self.files if f.split('.nii')[0] in valid_files]

        # Simple in-memory cache (huge speedup)
        self.cache = {}

        # Transforms (less aggressive → better R²)
        self.train_tf = Compose([
            RandomCrop((128,128,128), always_apply=True),
            ElasticTransform((0, 0.05), p=0.3),
            RandomRotate90((0,1), p=0.5),
            Normalize(always_apply=True)
        ])

        self.val_tf = Compose([
            CenterCrop((128,128,128), always_apply=True),
            Normalize(always_apply=True)
        ])

    def __len__(self):
        return len(self.files)

    def load_nifti(self, path):
        if path not in self.cache:
            self.cache[path] = nib.load(path).get_fdata()
        return self.cache[path]

    def __getitem__(self, idx):
        filename = self.files[idx]
        base = filename.split('.nii')[0]

        # Load volumes
        x = self.load_nifti(os.path.join(self.input_dir, filename))
        y = self.load_nifti(os.path.join(self.target_dir, base + '_seg.nii'))

        data = {'image': x, 'mask': y}

        if self.transform:
            data = self.train_tf(**data)
        else:
            data = self.val_tf(**data)

        x = data['image']
        y = data['mask']

        # Normalize (CRITICAL for R²)
        x = (x - x.mean()) / (x.std() + 1e-8)

        # Get regression target
        row = self.df[self.df['filenames'] == base]
        if len(row) == 0:
            raise ValueError(f"No label for {base}")

        y_adas = row['ADAS11'].values[0]

        return (
            torch.tensor(x, dtype=torch.float32),      # [D,H,W]
            torch.tensor(y, dtype=torch.long),         # segmentation
            torch.tensor(y_adas, dtype=torch.float32), # regression
            base
        )


# =====================
# Patient-level split (8-1-1)
# =====================

def create_patient_split(files, seed=42):
    patient_ids = list(set([f[8:16] for f in files]))

    train_p, temp_p = train_test_split(
        patient_ids, test_size=0.2, random_state=seed
    )

    val_p, test_p = train_test_split(
        temp_p, test_size=0.5, random_state=seed
    )

    return train_p, val_p, test_p


def split_files_by_patient(files, train_p, val_p, test_p):
    train, val, test = [], [], []

    for f in files:
        pid = f[8:16]
        if pid in train_p:
            train.append(f)
        elif pid in val_p:
            val.append(f)
        elif pid in test_p:
            test.append(f)

    return train, val, test


# =====================
# Data initialization
# =====================

def load_all_files():
    path = '/home/madar/Downloads/train_files5.data'
    with open(path, 'rb') as f:
        files = pickle.load(f)
    return files


def initialize_data():
    input_path = '/media/rajlab/sachin_data_1/userdata/daren/mri/'
    target_path = '/media/rajlab/sachin_data_1/userdata/daren/target/target_files/'
    df = pd.read_csv('cleaned_df_5_31.csv')
    return input_path, target_path, df


# =====================
# Dataloaders
# =====================

def get_ds_dl(batch_size=5, num_workers=8):
    input_path, target_path, df = initialize_data()
    all_files = load_all_files()

    train_p, val_p, test_p = create_patient_split(all_files)
    X_train, X_val, X_test = split_files_by_patient(
        all_files, train_p, val_p, test_p
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    ds_train = CogDataset3d(input_path, target_path, X_train, df, transform=True)
    ds_val   = CogDataset3d(input_path, target_path, X_val, df, transform=False)
    ds_test  = CogDataset3d(input_path, target_path, X_test, df, transform=False)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)

    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

    return ds_train, ds_val, ds_test, dl_train, dl_val, dl_test
