
# ============================================================
# 1. Bootstrap and experiment configuration
# ============================================================
# Install dependencies before running:
#   pip install -r requirements.txt

import os, sys, json, time, copy, random, warnings, re, glob, shutil
import tarfile, zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve, brier_score_loss)
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ---- Task setup ----
TASK_SETUP = "cn_vs_ad"          # "cn_vs_ad" | "cn_mci_ad"

# ---- Execution preset ----
PRESET = "colab_cpu"

PRESETS = {
    "colab_cpu": {
        "image_size": (48, 48, 48), "batch_size": 2,
        "epochs": 12, "patience": 5,
        "resnet_base": 16, "resnet_layers": (1, 1, 1, 1),
        "densenet_growth": 12, "densenet_layers": (3, 4, 6, 4),
        "fusion_dim": 128, "shared_fc": (128, 64),
    },
    "colab_gpu": {
        "image_size": (96, 96, 96), "batch_size": 4,
        "epochs": 30, "patience": 8,
        "resnet_base": 32, "resnet_layers": (2, 2, 2, 2),
        "densenet_growth": 16, "densenet_layers": (4, 6, 8, 6),
        "fusion_dim": 256, "shared_fc": (256, 128),
    },
}

CFG = {
    **PRESETS[PRESET],
    "preset": PRESET,
    "task_setup": TASK_SETUP,
    "experiment": "TXAI_reference_pipeline",
    "seed": 42,
    "lr": 1e-4, "weight_decay": 1e-4, "dropout": 0.5,
    "fusion_heads": 8,
    "outer_folds_max": 5,
    "val_frac_of_dev": 0.10 / 0.80,
    "clinical_mlp": (256, 128),
    "biomarker_mlp": (128, 64),
    "demographic_mlp": (64, 32),
    "lambda_diag": 1.0, "lambda_stage": 1.0,
    "lambda_prog": 0.5, "lambda_risk": 0.3,
    "mc_dropout_passes": 20, "ece_bins": 15,
    "ig_steps": 16, "xai_max_samples": 10,
    "shap_background": 20, "shap_explain": 5,
    "xavier_init": True,
}

OUT_ROOT = "/content/txai_trustworthy_outputs"
OUT_DIR = os.path.join(OUT_ROOT, f"{TASK_SETUP}_{PRESET}")
if os.path.isdir(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
CFG["output_dir"] = OUT_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("TXAI Reference Pipeline")
print("=" * 70)
print(f"Preset          : {PRESET}")
print(f"Task setup      : {TASK_SETUP}")
print(f"Image size      : {CFG['image_size']}")
print(f"Batch size      : {CFG['batch_size']}")
print(f"Epochs / fold   : {CFG['epochs']}")
print(f"Device          : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
print(f"Outputs         : {OUT_DIR}")
print("=" * 70)


# ============================================================
# 2. Data access and OASIS-1 extraction
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

OASIS_ARCHIVE = "/content/drive/MyDrive/oasis_cross-sectional_disc1.tar.gz"
DEST = "/content/oasis_data"
os.makedirs(DEST, exist_ok=True)

if not os.path.isdir(os.path.join(DEST, "disc1")):
    print(f"Extracting {OASIS_ARCHIVE} -> {DEST} ...")
    if OASIS_ARCHIVE.endswith(".zip"):
        with zipfile.ZipFile(OASIS_ARCHIVE) as z:
            z.extractall(DEST)
    elif OASIS_ARCHIVE.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(OASIS_ARCHIVE) as t:
            t.extractall(DEST)
    else:
        raise ValueError("Unsupported archive format")

ROOT = os.path.join(DEST, "disc1")
if not os.path.isdir(ROOT):
    hits = glob.glob(os.path.join(DEST, "**", "disc1"), recursive=True)
    if hits:
        ROOT = hits[0]
    else:
        raise RuntimeError(f"disc1 not found under {DEST}")
print("OASIS root:", ROOT)


# ============================================================
# 3. Manifest construction and task setup
# ============================================================
TAB_DIR = "/content/txai_data/tabular"
os.makedirs(TAB_DIR, exist_ok=True)
MANIFEST_CSV = "/content/txai_data/manifest.csv"


def parse_oasis_txt(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    text = open(path, "r", errors="ignore").read()
    fields = {}
    for k, pat in {
        "age":       r"Age:\s*([0-9]+)",
        "sex":       r"Sex:\s*([MF])",
        "education": r"Education:\s*([0-9]+)",
        "CDR":       r"CDR:\s*([0-9.]+)",
        "MMSE":      r"MMSE:\s*([0-9]+)",
    }.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                v = m.group(1).strip()
                fields[k] = (0 if v.upper() == "M" else 1) if k == "sex" \
                            else (float(v) if "." in v else int(v))
            except ValueError:
                pass
    return fields


def find_mri(subject_dir: str) -> str:
    for pat in [
        os.path.join(subject_dir, "PROCESSED", "MPRAGE", "T88_111", "*masked_gfc.hdr"),
        os.path.join(subject_dir, "PROCESSED", "MPRAGE", "T88_111", "*masked_gfc.img"),
        os.path.join(subject_dir, "RAW", "*.hdr"),
        os.path.join(subject_dir, "RAW", "*.img"),
    ]:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return ""


def build_manifest_raw() -> pd.DataFrame:
    """CDR is stored separately for label derivation but never fed to
    the model (avoids label leakage)."""
    rows = []
    participants = sorted(d for d in os.listdir(ROOT)
                          if d.startswith("OAS1_")
                          and os.path.isdir(os.path.join(ROOT, d)))
    for sid in participants:
        subj_dir = os.path.join(ROOT, sid)
        info = parse_oasis_txt(os.path.join(subj_dir, f"{sid}.txt"))
        cdr = info.get("CDR")
        if cdr is None: continue
        mri = find_mri(subj_dir)
        if not mri: continue

        cli_p = os.path.join(TAB_DIR, f"{sid}_clinical.json")
        dem_p = os.path.join(TAB_DIR, f"{sid}_demographic.json")
        with open(cli_p, "w") as f:
            json.dump({"MMSE": info.get("MMSE"), "ADAS_Cog": None}, f)
        with open(dem_p, "w") as f:
            json.dump({"age": info.get("age"), "sex": info.get("sex"),
                       "education": info.get("education")}, f)

        rows.append({
            "patient_id":  sid,
            "session_id":  f"{sid}_M00",
            "visit_month": 0,
            "cdr_raw":     float(cdr),
            "stage":       -1,
            "mri_path":    mri,
            "pet_path":    "",
            "clinical_json":    cli_p,
            "biomarker_json":   "",
            "demographic_json": dem_p,
        })
    return pd.DataFrame(rows)


def apply_task_setup(df: pd.DataFrame, setup: str) -> Tuple[pd.DataFrame, List[int]]:
    if setup == "cn_vs_ad":
        df = df[df["cdr_raw"] != 0.5].copy()
        df["diagnosis"] = (df["cdr_raw"] >= 1.0).astype(int)
        labels = [0, 1]
    elif setup == "cn_mci_ad":
        def cdr2dx(c):
            if c == 0: return 0
            if c == 0.5: return 1
            return 2
        df = df.copy()
        df["diagnosis"] = df["cdr_raw"].apply(cdr2dx).astype(int)
        labels = [0, 1, 2]
    else:
        raise ValueError(f"Unknown TASK_SETUP: {setup}")
    return df.reset_index(drop=True), labels


def build_progression_targets(df: pd.DataFrame) -> pd.DataFrame:
    STAGE_TO_INT = {"CN": 0, "EMCI": 1, "LMCI": 2, "AD": 3}
    def s2i(v):
        try: return int(v)
        except (TypeError, ValueError):
            return STAGE_TO_INT.get(str(v).strip().upper(), -1)
    rows = []
    for pid, g in df.groupby("patient_id"):
        g = g.sort_values("visit_month" if "visit_month" in g.columns else "session_id")
        seq = [s2i(s) for s in g["stage"].tolist()]
        seq = [s for s in seq if s >= 0]
        cn2e = int(any(a == 0 and b == 1 for a, b in zip(seq, seq[1:])))
        e2l  = int(any(a == 1 and b == 2 for a, b in zip(seq, seq[1:])))
        l2a  = int(any(a == 2 and b == 3 for a, b in zip(seq, seq[1:])))
        rows.append({
            "patient_id": pid,
            "prog_cn_emci": cn2e, "prog_emci_lmci": e2l, "prog_lmci_ad": l2a,
            "risk_target": float(np.mean([cn2e, e2l, l2a])),
        })
    return pd.DataFrame(rows)


df_manifest_raw = build_manifest_raw()
print(f"\nRaw OASIS participants with MRI and CDR: {len(df_manifest_raw)}")
print("Raw CDR distribution:")
print(df_manifest_raw["cdr_raw"].value_counts().sort_index().to_string())

df_manifest, DIAG_LABELS = apply_task_setup(df_manifest_raw, TASK_SETUP)
CFG["n_diag_classes"] = len(DIAG_LABELS)
CFG["n_stage_classes"] = 4

prog = build_progression_targets(df_manifest)
df_manifest = df_manifest.merge(prog, on="patient_id", how="left")
for c in ("prog_cn_emci", "prog_emci_lmci", "prog_lmci_ad"):
    df_manifest[c] = df_manifest[c].fillna(0).astype(int)
df_manifest["risk_target"] = df_manifest["risk_target"].fillna(0.0).astype(float)

df_manifest.to_csv(MANIFEST_CSV, index=False)
print(f"\nTask setup applied: {TASK_SETUP}")
print(f"Manifest           : {MANIFEST_CSV}")
print(f"Rows               : {len(df_manifest)}")
print("Diagnosis distribution:")
print(df_manifest["diagnosis"].value_counts().sort_index().to_string())
print("\nModality availability:")
for col, name in [("mri_path","MRI"), ("pet_path","PET"),
                  ("clinical_json","Clinical"), ("biomarker_json","Biomarker"),
                  ("demographic_json","Demographic")]:
    n = (df_manifest[col].astype(str) != "").sum()
    print(f"  {name:12s}: {n:3d} / {len(df_manifest)}")

class_counts = df_manifest["diagnosis"].value_counts()
min_class_count = int(class_counts.min())
n_splits_eff = max(2, min(CFG["outer_folds_max"], min_class_count))
CFG["outer_folds"] = n_splits_eff
print(f"\nSmallest class count : {min_class_count}")
print(f"Effective folds      : {n_splits_eff}")


# ============================================================
# 4. Dataset and preprocessing pipeline
# ============================================================
CLINICAL_KEYS  = ["MMSE", "ADAS_Cog"]
BIOMARKER_KEYS = ["AB42", "T_tau", "P_tau", "APOE"]
DEMO_KEYS      = ["age", "sex", "education"]

ACTIVE_MODALITIES = []
if (df_manifest["mri_path"].astype(str) != "").any():         ACTIVE_MODALITIES.append("mri")
if (df_manifest["pet_path"].astype(str) != "").any():         ACTIVE_MODALITIES.append("pet")
if (df_manifest["clinical_json"].astype(str) != "").any():    ACTIVE_MODALITIES.append("clinical")
if (df_manifest["biomarker_json"].astype(str) != "").any():   ACTIVE_MODALITIES.append("biomarker")
if (df_manifest["demographic_json"].astype(str) != "").any(): ACTIVE_MODALITIES.append("demographic")
print("\nActive modalities:", ACTIVE_MODALITIES)

TASKS_ACTIVE = {
    "diagnosis":   True,
    "staging":     bool((df_manifest["stage"] >= 0).any()),
    "progression": bool(
        (df_manifest["prog_cn_emci"].sum() +
         df_manifest["prog_emci_lmci"].sum() +
         df_manifest["prog_lmci_ad"].sum()) > 0
    ),
    "risk":        bool((df_manifest["risk_target"] > 0).any()),
}
print("\nTask availability for this run:")
for t, a in TASKS_ACTIVE.items():
    print(f"  {t:12s}: {'ACTIVE' if a else 'INACTIVE'}")


def load_nifti(p): return np.squeeze(np.asarray(nib.load(p).get_fdata(), dtype=np.float32))
def resize_volume(v, size):
    t = torch.from_numpy(v).float().unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=size, mode="trilinear",
                         align_corners=False).squeeze(0).squeeze(0).numpy()
def load_json(p):
    if not p or str(p) == "" or not os.path.exists(str(p)): return {}
    with open(p, "r") as f: return json.load(f)
def feat_vec(d, keys):
    out = []
    for k in keys:
        try: out.append(float(d.get(k, np.nan)))
        except (TypeError, ValueError): out.append(np.nan)
    return np.array(out, dtype=np.float32)
def modality_mask_row(row):
    return np.array([
        bool(row.get("mri_path"))         and os.path.exists(str(row["mri_path"])),
        bool(row.get("pet_path"))         and os.path.exists(str(row["pet_path"])),
        bool(row.get("clinical_json"))    and os.path.exists(str(row["clinical_json"])),
        bool(row.get("biomarker_json"))   and os.path.exists(str(row["biomarker_json"])),
        bool(row.get("demographic_json")) and os.path.exists(str(row["demographic_json"])),
    ], dtype=bool)


class TXAIDataset(Dataset):
    def __init__(self, df, image_size, augment=False, stats=None):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment
        self.stats = stats or {}
    def __len__(self): return len(self.df)
    def _img(self, path, key):
        if not path or not os.path.exists(str(path)): return None
        v = resize_volume(load_nifti(str(path)), self.image_size)
        s = self.stats.get(key, {"mean": 0.0, "std": 1.0})
        out = np.zeros_like(v, dtype=np.float32)
        fg = v != 0
        out[fg] = (v[fg] - s["mean"]) / (s["std"] + 1e-8)
        return out
    def _aug(self, v):
        if v is None: return None
        if random.random() < 0.5: v = v[::-1, :, :].copy()
        if random.random() < 0.5:
            k = random.randint(1, 3); v = np.rot90(v, k, axes=(1, 2)).copy()
        return v
    def _tab(self, path, keys, key):
        v = feat_vec(load_json(path), keys)
        s = self.stats.get(key, {})
        m = s.get("mean", np.zeros_like(v)); sd = s.get("std", np.ones_like(v))
        v = np.where(np.isnan(v), m, v)
        return ((v - m) / (sd + 1e-8)).astype(np.float32)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mri = self._img(row.get("mri_path"), "mri")
        pet = self._img(row.get("pet_path"), "pet")
        if self.augment:
            mri = self._aug(mri); pet = self._aug(pet)
        mri_t = torch.from_numpy(mri).float().unsqueeze(0) if mri is not None \
                else torch.zeros(1, *self.image_size)
        pet_t = torch.from_numpy(pet).float().unsqueeze(0) if pet is not None \
                else torch.zeros(1, *self.image_size)
        cli = torch.from_numpy(self._tab(row.get("clinical_json"),  CLINICAL_KEYS,  "clinical"))
        bio = torch.from_numpy(self._tab(row.get("biomarker_json"), BIOMARKER_KEYS, "biomarker"))
        dem = torch.from_numpy(self._tab(row.get("demographic_json"), DEMO_KEYS,    "demographic"))
        mask = torch.tensor(modality_mask_row(row), dtype=torch.bool)
        return {
            "mri": mri_t, "pet": pet_t,
            "clinical": cli, "biomarker": bio, "demographic": dem,
            "mask": mask,
            "diagnosis": torch.tensor(int(row["diagnosis"]), dtype=torch.long),
            "stage":     torch.tensor(int(row.get("stage", -1)), dtype=torch.long),
            "prog":      torch.tensor([int(row.get("prog_cn_emci", 0)),
                                        int(row.get("prog_emci_lmci", 0)),
                                        int(row.get("prog_lmci_ad", 0))], dtype=torch.float32),
            "risk":      torch.tensor(float(row.get("risk_target", 0.0)), dtype=torch.float32),
            "patient_id": row["patient_id"],
        }


def compute_train_stats(train_df):
    stats = {}
    def img_stats(key, col):
        s = sq = 0.0; n = 0
        for _, r in train_df.iterrows():
            p = r.get(col, "")
            if not p or not os.path.exists(str(p)): continue
            v = resize_volume(load_nifti(str(p)), CFG["image_size"])
            fg = v[v != 0].astype(np.float64)
            s += fg.sum(); sq += (fg ** 2).sum(); n += fg.size
        if n:
            m = s / n; var = max(sq / n - m ** 2, 1e-12)
            stats[key] = {"mean": float(m), "std": float(np.sqrt(var))}
        else:
            stats[key] = {"mean": 0.0, "std": 1.0}
    img_stats("mri", "mri_path"); img_stats("pet", "pet_path")
    for key, col, keys in [("clinical","clinical_json",CLINICAL_KEYS),
                           ("biomarker","biomarker_json",BIOMARKER_KEYS),
                           ("demographic","demographic_json",DEMO_KEYS)]:
        V = np.stack([feat_vec(load_json(r.get(col, "")), keys)
                      for _, r in train_df.iterrows()], 0)
        with np.errstate(all="ignore"):
            m = np.nanmean(V, axis=0); sd = np.nanstd(V, axis=0)
        m = np.where(np.isnan(m), 0.0, m)
        sd = np.where(np.isnan(sd) | (sd == 0), 1.0, sd)
        stats[key] = {"mean": m, "std": sd}
    return stats


# ============================================================
# 5. Multimodal TXAI model
# ============================================================
def xavier_init_module(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            try:
                nn.init.xavier_uniform_(m.weight)
            except Exception:
                pass
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)


class Bottleneck3D(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None: identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet3DBackbone(nn.Module):
    def __init__(self, in_ch=1, out_dim=128, base_width=16, layers=(1,1,1,1)):
        super().__init__()
        self.inplanes = base_width
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, base_width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_width), nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1))
        self.layer1 = self._mk(base_width,     layers[0], 1)
        self.layer2 = self._mk(base_width * 2, layers[1], 2)
        self.layer3 = self._mk(base_width * 4, layers[2], 2)
        self.layer4 = self._mk(base_width * 8, layers[3], 2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(base_width * 8 * Bottleneck3D.expansion, out_dim)
        self.gradcam_target = self.layer4[-1]
    def _mk(self, planes, blocks, stride):
        ds = None
        if stride != 1 or self.inplanes != planes * Bottleneck3D.expansion:
            ds = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * Bottleneck3D.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(planes * Bottleneck3D.expansion))
        layers = [Bottleneck3D(self.inplanes, planes, stride, ds)]
        self.inplanes = planes * Bottleneck3D.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck3D(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.proj(self.pool(x).flatten(1))


class DenseLayer3D(nn.Module):
    def __init__(self, in_ch, growth):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, growth, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(growth); self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return torch.cat([x, self.relu(self.bn(self.conv(x)))], dim=1)


class DenseBlock3D(nn.Module):
    def __init__(self, in_ch, growth, n):
        super().__init__()
        layers, c = [], in_ch
        for _ in range(n): layers.append(DenseLayer3D(c, growth)); c += growth
        self.layers = nn.Sequential(*layers); self.out_ch = c
    def forward(self, x): return self.layers(x)


class Transition3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.pool = nn.AvgPool3d(2, stride=2)
    def forward(self, x): return self.pool(self.bn(self.conv(x)))


class DenseNet3DStyle(nn.Module):
    """DenseNet-style 3D backbone for PET."""
    def __init__(self, in_ch=1, out_dim=128, growth=12, layers=(3,4,6,4)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1))
        self.d1 = DenseBlock3D(32, growth, layers[0])
        self.t1 = Transition3D(self.d1.out_ch, self.d1.out_ch // 2)
        self.d2 = DenseBlock3D(self.d1.out_ch // 2, growth, layers[1])
        self.t2 = Transition3D(self.d2.out_ch, self.d2.out_ch // 2)
        self.d3 = DenseBlock3D(self.d2.out_ch // 2, growth, layers[2])
        self.t3 = Transition3D(self.d3.out_ch, self.d3.out_ch // 2)
        self.d4 = DenseBlock3D(self.d3.out_ch // 2, growth, layers[3])
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(self.d4.out_ch, out_dim)
    def forward(self, x):
        x = self.stem(x)
        x = self.t1(self.d1(x)); x = self.t2(self.d2(x))
        x = self.t3(self.d3(x)); x = self.d4(x)
        return self.proj(self.pool(x).flatten(1))


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.5):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class MCSAFusion(nn.Module):
    def __init__(self, n_mod, dim, heads=8, dropout=0.5):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim); self.out = nn.Linear(dim, dim)
        self.mod_attn = nn.Linear(dim, 1)
        self.drop = nn.Dropout(dropout)
        self.missing = nn.Parameter(torch.zeros(1, n_mod, dim))
        nn.init.normal_(self.missing, std=0.02)
    def forward(self, feats, mask=None):
        stack = torch.stack(feats, dim=1)
        B, M, D = stack.shape
        if mask is not None:
            m = mask.view(B, M, 1).float()
            stack = stack * m + self.missing * (1 - m)
        H = self.heads; Dh = D // H
        Q = self.q(stack).view(B, M, H, Dh).transpose(1, 2)
        K = self.k(stack).view(B, M, H, Dh).transpose(1, 2)
        V = self.v(stack).view(B, M, H, Dh).transpose(1, 2)
        s = Q @ K.transpose(-2, -1) / (Dh ** 0.5)
        if mask is not None:
            s = s.masked_fill((~mask).view(B, 1, 1, M).expand_as(s), -1e9)
        a = torch.softmax(s, dim=-1)
        o = self.out(self.drop((a @ V).transpose(1, 2).reshape(B, M, D)))
        ms = self.mod_attn(o).squeeze(-1)
        if mask is not None:
            ms = ms.masked_fill(~mask, -1e9)
        alpha = torch.softmax(ms, dim=-1)
        return (alpha.unsqueeze(-1) * o).sum(1), alpha, a


class TXAI(nn.Module):
    def __init__(self):
        super().__init__()
        fd = CFG["fusion_dim"]
        self.active = list(ACTIVE_MODALITIES)
        self.has_mri = "mri" in self.active
        self.has_pet = "pet" in self.active
        self.has_cli = "clinical" in self.active
        self.has_bio = "biomarker" in self.active
        self.has_dem = "demographic" in self.active
        if self.has_mri:
            self.mri_enc = ResNet3DBackbone(1, fd, CFG["resnet_base"], CFG["resnet_layers"])
        if self.has_pet:
            self.pet_enc = DenseNet3DStyle(1, fd, CFG["densenet_growth"], CFG["densenet_layers"])
        if self.has_cli:
            self.cli_enc = MLPEncoder(len(CLINICAL_KEYS),  CFG["clinical_mlp"],    fd)
        if self.has_bio:
            self.bio_enc = MLPEncoder(len(BIOMARKER_KEYS), CFG["biomarker_mlp"],   fd)
        if self.has_dem:
            self.dem_enc = MLPEncoder(len(DEMO_KEYS),      CFG["demographic_mlp"], fd)
        self.fusion = MCSAFusion(5, fd, CFG["fusion_heads"], CFG["dropout"])
        self.shared = nn.Sequential(
            nn.Linear(fd, CFG["shared_fc"][0]), nn.ReLU(inplace=True),
            nn.Dropout(CFG["dropout"]),
            nn.Linear(CFG["shared_fc"][0], CFG["shared_fc"][1]), nn.ReLU(inplace=True),
            nn.Dropout(CFG["dropout"]))
        sf = CFG["shared_fc"][1]
        self.head_diag  = nn.Linear(sf, CFG["n_diag_classes"])
        self.head_stage = nn.Linear(sf, CFG["n_stage_classes"])
        self.head_prog  = nn.Linear(sf, 3)
        self.head_risk  = nn.Linear(sf, 1)
        if CFG.get("xavier_init", True):
            xavier_init_module(self)
    def _zero(self, batch):
        return torch.zeros(batch["diagnosis"].size(0), CFG["fusion_dim"],
                           device=batch["diagnosis"].device)
    def forward(self, batch):
        fmri = self.mri_enc(batch["mri"]) if self.has_mri else self._zero(batch)
        fpet = self.pet_enc(batch["pet"]) if self.has_pet else self._zero(batch)
        fcli = self.cli_enc(batch["clinical"])  if self.has_cli else self._zero(batch)
        fbio = self.bio_enc(batch["biomarker"]) if self.has_bio else self._zero(batch)
        fdem = self.dem_enc(batch["demographic"]) if self.has_dem else self._zero(batch)
        fused, alpha, attn = self.fusion([fmri, fpet, fcli, fbio, fdem], mask=batch["mask"])
        h = self.shared(fused)
        return {"diag": self.head_diag(h), "stage": self.head_stage(h),
                "prog": self.head_prog(h), "risk": self.head_risk(h).squeeze(-1),
                "alpha": alpha, "attn": attn, "fused": fused}


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ============================================================
# 6. Losses, metrics, and training utilities
# ============================================================
def multitask_loss(out, batch):
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    l_diag = ce(out["diag"], batch["diagnosis"])
    sm = batch["stage"] >= 0
    l_stage = ce(out["stage"][sm], batch["stage"][sm]) if sm.any() \
              else torch.tensor(0.0, device=out["stage"].device)
    pt = batch["prog"]
    l_prog = mse(out["prog"], pt) if pt.std(dim=0).min() > 0 \
             else torch.tensor(0.0, device=out["prog"].device)
    l_risk = mse(out["risk"], batch["risk"]) if batch["risk"].std() > 0 \
             else torch.tensor(0.0, device=out["risk"].device)
    total = (CFG["lambda_diag"] * l_diag + CFG["lambda_stage"] * l_stage +
             CFG["lambda_prog"] * l_prog + CFG["lambda_risk"] * l_risk)
    return total, {"diag": float(l_diag), "stage": float(l_stage),
                   "prog": float(l_prog), "risk": float(l_risk)}


def _safe_auc(yt, yp, labels):
    if len(np.unique(yt)) < 2:
        return float("nan")
    try:
        if len(labels) == 2:
            return float(roc_auc_score(yt, yp[:, 1]))
        return float(roc_auc_score(yt, yp, multi_class="ovr",
                                    average="macro", labels=labels))
    except ValueError:
        return float("nan")


def run_epoch(model, loader, opt=None, train=True):
    model.train(train)
    tot = {"loss": 0.0, "diag": 0.0, "stage": 0.0, "prog": 0.0, "risk": 0.0}
    n = 0
    diag_t, diag_p = [], []
    stage_t, stage_p = [], []
    for b in loader:
        for k in ("mri", "pet", "clinical", "biomarker", "demographic"):
            b[k] = b[k].to(DEVICE, non_blocking=True)
        b["mask"] = b["mask"].to(DEVICE)
        for k in ("diagnosis", "stage", "prog", "risk"):
            b[k] = b[k].to(DEVICE)
        with torch.set_grad_enabled(train):
            out = model(b)
            loss, parts = multitask_loss(out, b)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        bs = b["diagnosis"].size(0); n += bs
        tot["loss"] += float(loss) * bs
        for k in ("diag", "stage", "prog", "risk"): tot[k] += parts[k] * bs
        diag_t.append(b["diagnosis"].cpu().numpy())
        diag_p.append(torch.softmax(out["diag"].detach(), 1).cpu().numpy())
        sm = b["stage"] >= 0
        if sm.any():
            stage_t.append(b["stage"][sm].cpu().numpy())
            stage_p.append(out["stage"][sm].argmax(-1).detach().cpu().numpy())

    yt = np.concatenate(diag_t)
    yp = np.concatenate(diag_p)
    ypred = yp.argmax(1)
    metrics = {f"loss_{k}": tot[k] / max(n, 1) for k in tot}
    metrics["diag_acc"]  = accuracy_score(yt, ypred)
    metrics["diag_prec"] = precision_score(yt, ypred, labels=DIAG_LABELS,
                                           average="macro", zero_division=0)
    metrics["diag_rec"]  = recall_score(yt, ypred, labels=DIAG_LABELS,
                                        average="macro", zero_division=0)
    metrics["diag_f1"]   = f1_score(yt, ypred, labels=DIAG_LABELS,
                                    average="macro", zero_division=0)
    metrics["diag_auc"]  = _safe_auc(yt, yp, DIAG_LABELS)
    if stage_t:
        metrics["stage_acc"] = accuracy_score(np.concatenate(stage_t),
                                              np.concatenate(stage_p))
    else:
        metrics["stage_acc"] = float("nan")
    return metrics, yt, yp


# ============================================================
# 7. Trustworthy-AI evaluation
# ============================================================
def enable_mc_dropout(m):
    m.eval()
    for mod in m.modules():
        if isinstance(mod, nn.Dropout): mod.train()


def mc_dropout(model, loader, T=20):
    enable_mc_dropout(model)
    means, vars_, trues = [], [], []
    with torch.no_grad():
        for b in loader:
            for k in ("mri", "pet", "clinical", "biomarker", "demographic"):
                b[k] = b[k].to(DEVICE)
            b["mask"] = b["mask"].to(DEVICE)
            ps = []
            for _ in range(T):
                o = model(b)
                ps.append(torch.softmax(o["diag"], 1).unsqueeze(0))
            ps = torch.cat(ps, 0)
            means.append(ps.mean(0).cpu().numpy())
            vars_.append(ps.var(0).sum(-1).cpu().numpy())
            trues.append(b["diagnosis"].cpu().numpy())
    model.eval()
    return np.concatenate(means), np.concatenate(vars_), np.concatenate(trues)


def ece(probs, labels, bins=15):
    conf = probs.max(1); pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += (m.sum() / len(labels)) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def fit_temperature(val_logits, val_labels):
    T = nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=30)
    ce = nn.CrossEntropyLoss()
    L = torch.from_numpy(val_logits).float()
    Y = torch.from_numpy(val_labels).long()
    def closure():
        opt.zero_grad()
        loss = ce(L / T.clamp(min=1e-3), Y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.item())


def softmax_np(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def trustworthy(model, test_ld, val_ld):
    r = {}
    model.eval()
    vl, vlab = [], []
    with torch.no_grad():
        for b in val_ld:
            for k in ("mri", "pet", "clinical", "biomarker", "demographic"):
                b[k] = b[k].to(DEVICE)
            b["mask"] = b["mask"].to(DEVICE)
            o = model(b)
            vl.append(o["diag"].cpu().numpy())
            vlab.append(b["diagnosis"].cpu().numpy())
    T = fit_temperature(np.concatenate(vl), np.concatenate(vlab))
    r["temperature"] = T
    mean_p, var_u, Y = mc_dropout(model, test_ld, T=CFG["mc_dropout_passes"])
    r["uncertainty_mean"] = float(var_u.mean())
    r["uncertainty_std"] = float(var_u.std())

    logits, labs = [], []
    model.eval()
    with torch.no_grad():
        for b in test_ld:
            for k in ("mri", "pet", "clinical", "biomarker", "demographic"):
                b[k] = b[k].to(DEVICE)
            b["mask"] = b["mask"].to(DEVICE)
            o = model(b)
            logits.append(o["diag"].cpu().numpy())
            labs.append(b["diagnosis"].cpu().numpy())
    L = np.concatenate(logits); Yl = np.concatenate(labs)
    pb = softmax_np(L); pa = softmax_np(L / T)
    r["ece_before"] = ece(pb, Yl, CFG["ece_bins"])
    r["ece_after"]  = ece(pa, Yl, CFG["ece_bins"])
    r["brier_before"] = float(np.mean([brier_score_loss((Yl == c).astype(int), pb[:, c])
                                       for c in range(CFG["n_diag_classes"])]))
    r["brier_after"] = float(np.mean([brier_score_loss((Yl == c).astype(int), pa[:, c])
                                      for c in range(CFG["n_diag_classes"])]))

    order = np.argsort(var_u)
    r["selective_prediction"] = {
        float(f): float(accuracy_score(Yl[order[:max(int(round(f * len(order))), 1)]],
                                       mean_p[order[:max(int(round(f * len(order))), 1)]].argmax(1)))
        for f in np.linspace(0.1, 1.0, 10)
    }
    conf = mean_p.max(1)
    unc_n = 1.0 - var_u / (var_u.max() + 1e-8)
    cal_n = 1.0 - min(r["ece_after"], 1.0)
    r["reliability_index_mean"] = float(np.mean(
        0.30 * unc_n + 0.30 * conf + 0.20 * cal_n + 0.20 * 1.0))
    return r


def robustness(model, test_ld, base):
    res = {}
    for kind, levels in {"gaussian_noise": [0.05, 0.10],
                         "intensity": [0.9, 0.8],
                         "spatial_shift": [1, 2]}.items():
        res[kind] = []
        for lvl in levels:
            yt, yp = [], []
            model.eval()
            with torch.no_grad():
                for b in test_ld:
                    xm = b["mri"].clone(); xp = b["pet"].clone()
                    if kind == "gaussian_noise":
                        xm = xm + lvl * torch.randn_like(xm)
                        xp = xp + lvl * torch.randn_like(xp)
                    elif kind == "intensity":
                        xm = xm * lvl; xp = xp * lvl
                    elif kind == "spatial_shift":
                        xm = torch.roll(xm, lvl, dims=2)
                        xp = torch.roll(xp, lvl, dims=2)
                    b2 = dict(b)
                    b2["mri"] = xm.to(DEVICE); b2["pet"] = xp.to(DEVICE)
                    for k in ("clinical", "biomarker", "demographic"):
                        b2[k] = b2[k].to(DEVICE)
                    b2["mask"] = b2["mask"].to(DEVICE)
                    o = model(b2)
                    yt.append(b["diagnosis"].numpy())
                    yp.append(torch.softmax(o["diag"], 1).cpu().numpy())
            yt = np.concatenate(yt); yp = np.concatenate(yp)
            res[kind].append({
                "level": float(lvl),
                "accuracy": float(accuracy_score(yt, yp.argmax(1))),
                "f1": float(f1_score(yt, yp.argmax(1), average="macro", zero_division=0)),
                "deg_acc": float(base.get("diag_acc", 0) - accuracy_score(yt, yp.argmax(1))),
            })
    return res


# ============================================================
# 8. Explainability (XAI)
# ============================================================
def integrated_gradients(model, batch, target, steps=16):
    if not model.has_mri: return None
    x = batch["mri"].clone().detach().requires_grad_(True)
    base = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for a in torch.linspace(0, 1, steps, device=x.device):
        xa = (base + a * (x - base)).detach().requires_grad_(True)
        b2 = dict(batch); b2["mri"] = xa
        out = model(b2)
        s = out["diag"][:, target].sum()
        model.zero_grad(); s.backward(retain_graph=False)
        total = total + xa.grad.detach()
    return ((x - base) * total / steps).detach().cpu().numpy()


def grad_cam_3d(model, batch, target=None):
    if not model.has_mri: return None
    acts, grds = {}, {}
    tgt = model.mri_enc.gradcam_target
    def fwd(_, __, out): acts["a"] = out
    def bwd(_, gi, go): grds["g"] = go[0]
    h1 = tgt.register_forward_hook(fwd)
    h2 = tgt.register_full_backward_hook(bwd)
    try:
        out = model(batch)
        if target is None: target = out["diag"].argmax(1)
        s = out["diag"].gather(1, target.view(-1, 1)).sum()
        model.zero_grad(); s.backward()
        a = acts["a"]; g = grds["g"]
        w = g.mean(dim=(2, 3, 4), keepdim=True)
        return F.relu((w * a).sum(1)).detach().cpu().numpy()
    finally:
        h1.remove(); h2.remove()


def compute_shap_tabular(model, test_ld, n_bg=None, n_ex=None):
    n_bg = n_bg or CFG["shap_background"]
    n_ex = n_ex or CFG["shap_explain"]
    try:
        import shap
    except ImportError:
        return {"error": "shap not installed"}

    model.eval()
    tabs, ref = [], None
    with torch.no_grad():
        for b in test_ld:
            for k in ("mri", "pet", "clinical", "biomarker", "demographic"):
                b[k] = b[k].to(DEVICE)
            b["mask"] = b["mask"].to(DEVICE)
            if ref is None:
                ref = {k: v.detach().clone() for k, v in b.items()
                       if isinstance(v, torch.Tensor)}
            x = torch.cat([b["clinical"], b["biomarker"], b["demographic"]], dim=1)
            tabs.append(x.cpu())

    X = torch.cat(tabs, 0)
    if X.size(0) < 2:
        return {"error": "insufficient samples"}
    n_bg = min(n_bg, X.size(0)); n_ex = min(n_ex, X.size(0))
    bg = X[:n_bg].to(DEVICE); ex = X[:n_ex].to(DEVICE)
    nc, nb = len(CLINICAL_KEYS), len(BIOMARKER_KEYS)

    class Wrap(nn.Module):
        def __init__(self, model, ref):
            super().__init__(); self.model = model; self.ref = ref
        def forward(self, x):
            B = x.size(0)
            batch = {}
            for k, v in self.ref.items():
                if isinstance(v, torch.Tensor):
                    if v.size(0) == 1: batch[k] = v.expand(B, *v.shape[1:])
                    else: batch[k] = v[:1].expand(B, *v.shape[1:])
            batch["clinical"]    = x[:, :nc]
            batch["biomarker"]   = x[:, nc:nc + nb]
            batch["demographic"] = x[:, nc + nb:]
            return self.model(batch)["diag"]

    wrapped = Wrap(model, ref).to(DEVICE).eval()
    try:
        explainer = shap.GradientExplainer(wrapped, bg)
        sv = explainer.shap_values(ex)
        if isinstance(sv, list):
            mean_abs = np.stack([np.abs(s) for s in sv], 0).mean(axis=(0, 1))
        elif sv.ndim == 3:
            mean_abs = np.abs(sv).mean(axis=(0, 1))
        else:
            mean_abs = np.abs(sv).mean(axis=0)
        names = ([f"clinical.{k}" for k in CLINICAL_KEYS] +
                 [f"biomarker.{k}" for k in BIOMARKER_KEYS] +
                 [f"demographic.{k}" for k in DEMO_KEYS])
        order = np.argsort(-mean_abs)
        ranking = [{"feature": names[i], "mean_abs_shap": float(mean_abs[i])}
                   for i in order]
        return {"feature_names": names,
                "mean_abs_shap": mean_abs.tolist(),
                "ranking": ranking,
                "n_explained": int(n_ex),
                "n_background": int(n_bg)}
    except Exception as e:
        return {"error": f"SHAP failed: {type(e).__name__}: {e}"}


def run_xai(model, test_ld, n_max=10):
    model.eval()
    alphas, ys = [], []
    ig_means, cam_means = [], []
    seen = 0
    for b in test_ld:
        for k in ("mri", "pet", "clinical", "biomarker", "demographic"):
            b[k] = b[k].to(DEVICE)
        b["mask"] = b["mask"].to(DEVICE)
        with torch.no_grad():
            o = model(b)
        pred = o["diag"].argmax(1)
        alphas.append(o["alpha"].cpu().numpy())
        ys.append(b["diagnosis"].cpu().numpy())
        for i in range(b["diagnosis"].size(0)):
            if seen >= n_max: break
            b1 = {k: (v[i:i + 1] if isinstance(v, torch.Tensor) else v)
                  for k, v in b.items()}
            try:
                ig_means.append(float(np.nanmean(integrated_gradients(
                    model, b1, int(pred[i]), steps=CFG["ig_steps"]))))
            except Exception:
                ig_means.append(float("nan"))
            try:
                cam = grad_cam_3d(model, b1, int(pred[i]))
                cam_means.append(float(cam.mean()) if cam is not None else float("nan"))
            except Exception:
                cam_means.append(float("nan"))
            seen += 1

    alphas = np.concatenate(alphas); ys = np.concatenate(ys)
    cons = {}
    for c in np.unique(ys):
        a = alphas[ys == c]
        if len(a) > 1:
            cons[int(c)] = float(1.0 - np.mean(np.std(a, axis=0)) / (np.mean(a) + 1e-8))
    shap_res = compute_shap_tabular(model, test_ld)
    return {"alpha_mean": alphas.mean(0).tolist(),
            "alpha_std": alphas.std(0).tolist(),
            "attention_consistency": cons,
            "ig_mean": float(np.nanmean(ig_means)) if ig_means else float("nan"),
            "gradcam_mean": float(np.nanmean(cam_means)) if cam_means else float("nan"),
            "n_samples": int(seen),
            "shap": shap_res}


# ============================================================
# 9. Cross-validation and experiment entry point
# ============================================================
def safe_stratified_split(df, test_size, random_state):
    y = df["diagnosis"].values
    classes, counts = np.unique(y, return_counts=True)
    n_classes_present = len(classes)
    min_count = int(counts.min())
    if min_count < 2:
        print(f"  [warn] smallest class has only {min_count} sample(s) "
              f"in dev; falling back to non-stratified split")
        test_size = max(1, min(test_size, len(df) - 1))
        return train_test_split(df, test_size=test_size,
                                random_state=random_state, shuffle=True)
    test_size = max(int(test_size), n_classes_present)
    test_size = min(test_size, len(df) - 1)
    return train_test_split(df, test_size=test_size,
                            stratify=df["diagnosis"],
                            random_state=random_state)


def make_folds(df, k, seed):
    pid = df.groupby("patient_id").first().reset_index()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(pid, pid["diagnosis"])), pid


def main():
    def is_bl(s):
        s = str(s).lower()
        return ("baseline" in s) or s.endswith("bl") or ("m00" in s) or (s == "0")
    bm = df_manifest["session_id"].apply(is_bl)
    df_base = df_manifest[bm].copy().reset_index(drop=True) \
              if bm.sum() else df_manifest.groupby("patient_id", as_index=False).first()
    print(f"\nBaseline patients: {len(df_base)}")
    print(df_base["diagnosis"].value_counts().sort_index().to_string())
    print(f"Diagnosis labels  : {DIAG_LABELS}")

    probe = TXAI()
    actual = count_params(probe)
    print(f"\nModel trainable parameters: {actual / 1e6:.3f} M")
    del probe

    folds, pid_df = make_folds(df_base, CFG["outer_folds"], CFG["seed"])
    fold_rows = []
    oof_ids, oof_t, oof_p = [], [], []

    for fi, (dev_idx, test_idx) in enumerate(folds, start=1):
        seed = CFG["seed"] + fi
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

        dev_pids  = pid_df.iloc[dev_idx]["patient_id"].tolist()
        test_pids = pid_df.iloc[test_idx]["patient_id"].tolist()
        dev_df  = df_base[df_base["patient_id"].isin(dev_pids)].reset_index(drop=True)
        test_df = df_base[df_base["patient_id"].isin(test_pids)].reset_index(drop=True)

        n_val = max(int(round(CFG["val_frac_of_dev"] * len(dev_df))), 1)
        n_val = min(n_val, len(dev_df) - 1)
        train_df, val_df = safe_stratified_split(dev_df, test_size=n_val,
                                                 random_state=seed)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        for a, b in [(train_df, val_df), (train_df, test_df), (val_df, test_df)]:
            assert not (set(a["patient_id"]) & set(b["patient_id"])), "Patient overlap!"

        print(f"\nFold {fi}: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
        stats = compute_train_stats(train_df)
        tr_ds = TXAIDataset(train_df, CFG["image_size"], True, stats)
        va_ds = TXAIDataset(val_df, CFG["image_size"], False, stats)
        te_ds = TXAIDataset(test_df, CFG["image_size"], False, stats)
        tr_ld = DataLoader(tr_ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=0)
        va_ld = DataLoader(va_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=0)
        te_ld = DataLoader(te_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

        model = TXAI().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"],
                               weight_decay=CFG["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=3)

        best_v = float("inf"); best_state = None; best_ep = -1; bad = 0
        for ep in range(1, CFG["epochs"] + 1):
            t0 = time.time()
            tr, _, _ = run_epoch(model, tr_ld, opt, True)
            va, _, _ = run_epoch(model, va_ld, None, False)
            sched.step(va["loss_loss"])
            if va["loss_loss"] < best_v:
                best_v = va["loss_loss"]
                best_state = copy.deepcopy(model.state_dict())
                best_ep = ep; bad = 0
            else:
                bad += 1
            print(f"  ep {ep:3d} | tr={tr['loss_loss']:.4f} va={va['loss_loss']:.4f} "
                  f"| va_diag_acc={va.get('diag_acc', 0):.3f} | best_ep={best_ep} "
                  f"bad={bad} ({time.time() - t0:.1f}s)")
            if bad >= CFG["patience"]:
                print("  early stop"); break

        model.load_state_dict(best_state)
        te, yt, yp = run_epoch(model, te_ld, None, False)
        print(f"  [TEST] acc={te['diag_acc']:.4f} f1={te['diag_f1']:.4f} "
              f"auc={te['diag_auc']:.4f} stage_acc={te['stage_acc']}")

        try:
            tw = trustworthy(model, te_ld, va_ld)
            tw["robustness"] = robustness(model, te_ld, te)
            tw["active_tasks"] = TASKS_ACTIVE
            with open(os.path.join(OUT_DIR, f"trustworthy_fold{fi}.json"), "w") as f:
                json.dump(tw, f, indent=2, default=float)
        except Exception as e:
            print(f"  [TW] skipped: {e}")

        try:
            xai = run_xai(model, te_ld, n_max=CFG["xai_max_samples"])
            xai["active_tasks"] = TASKS_ACTIVE
            with open(os.path.join(OUT_DIR, f"xai_fold{fi}.json"), "w") as f:
                json.dump(xai, f, indent=2, default=float)
        except Exception as e:
            print(f"  [XAI] skipped: {e}")

        model.eval()
        with torch.no_grad():
            for b in te_ld:
                for k in ("mri", "pet", "clinical", "biomarker", "demographic"):
                    b[k] = b[k].to(DEVICE)
                b["mask"] = b["mask"].to(DEVICE)
                o = model(b)
                oof_p.append(torch.softmax(o["diag"], 1).cpu().numpy())
                oof_t.append(b["diagnosis"].cpu().numpy())
                oof_ids.extend(b["patient_id"])

        fold_rows.append({
            "fold": fi, "best_epoch": best_ep, "best_val_loss": best_v,
            **{k: te.get(k, float("nan")) for k in
               ["diag_acc", "diag_prec", "diag_rec", "diag_f1", "diag_auc", "stage_acc"]}
        })
        torch.save(best_state, os.path.join(OUT_DIR, f"txai_fold{fi}.pt"))

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(OUT_DIR, "fold_metrics.csv"), index=False)

    yt = np.concatenate(oof_t)
    yp = np.concatenate(oof_p)
    ypred = yp.argmax(1)
    pooled = {
        "task_setup":     TASK_SETUP,
        "n_patients":     int(len(yt)),
        "diag_accuracy":  float(accuracy_score(yt, ypred)),
        "diag_precision": float(precision_score(yt, ypred, labels=DIAG_LABELS,
                                                average="macro", zero_division=0)),
        "diag_recall":    float(recall_score(yt, ypred, labels=DIAG_LABELS,
                                             average="macro", zero_division=0)),
        "diag_f1":        float(f1_score(yt, ypred, labels=DIAG_LABELS,
                                         average="macro", zero_division=0)),
        "diag_auc":       _safe_auc(yt, yp, DIAG_LABELS),
        "confusion":      confusion_matrix(yt, ypred, labels=DIAG_LABELS).tolist(),
        "active_tasks":   TASKS_ACTIVE,
    }
    with open(os.path.join(OUT_DIR, "pooled_metrics.json"), "w") as f:
        json.dump(pooled, f, indent=2, default=float)
    pd.DataFrame({"patient_id": oof_ids, "y_true": yt,
                  "prob_class_0": yp[:, 0],
                  "prob_class_1": yp[:, 1] if yp.shape[1] > 1 else np.nan,
                  "prob_class_2": yp[:, 2] if yp.shape[1] > 2 else np.nan}
                 ).to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False)

    print("\n" + "=" * 70)
    print("CROSS-VALIDATED TEST (mean ± std across folds)")
    print("=" * 70)
    for m in ["diag_acc", "diag_prec", "diag_rec", "diag_f1", "diag_auc", "stage_acc"]:
        if m in fold_df:
            print(f"  {m:>12s}: {fold_df[m].mean():.4f} ± {fold_df[m].std():.4f}")

    print("\n" + "=" * 70)
    print("POOLED OUT-OF-FOLD")
    print("=" * 70)
    for k, v in pooled.items():
        if k not in ("confusion", "active_tasks"):
            print(f"  {k:>15s}: {v}")
    print(f"  confusion     : {pooled['confusion']}")
    print(f"  active tasks  : {pooled['active_tasks']}")

    try:
        plt.figure(figsize=(6, 5))
        if len(DIAG_LABELS) == 2:
            fpr, tpr, _ = roc_curve(yt, yp[:, 1])
            plt.plot(fpr, tpr, label=f"CN vs AD (AUC={pooled['diag_auc']:.3f})")
        else:
            for c, name in enumerate(["CN", "MCI", "AD"]):
                fpr, tpr, _ = roc_curve((yt == c).astype(int), yp[:, c])
                plt.plot(fpr, tpr, label=name)
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"Pooled OOF ROC — {TASK_SETUP} (OASIS-1)")
        plt.legend(loc="lower right"); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "pooled_roc.png"), dpi=150); plt.close()
    except Exception as e:
        print("ROC plot skipped:", e)

    with open(os.path.join(OUT_DIR, "experiment_config.json"), "w") as f:
        json.dump({**CFG, "device": str(DEVICE),
                   "active_modalities": ACTIVE_MODALITIES,
                   "active_tasks": TASKS_ACTIVE,
                   "diag_labels": DIAG_LABELS}, f, indent=2, default=str)

    print("\nArtifacts saved to:", OUT_DIR)
    for f in sorted(os.listdir(OUT_DIR)):
        print(" ", f)


if __name__ == "__main__":
    main()