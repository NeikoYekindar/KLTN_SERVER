"""
=============================================================================
TCN Weather Forecasting - Hard Targets Training Script
=============================================================================
Xử lý riêng các chỉ số khó dự báo:
  - wind_direction : chuyển sang sin/cos (góc tròn)
  - precipitation  : regression chuẩn (lượng mưa mm)
  - rain_probability: sigmoid 0-1 (xác suất mưa)
  - gust_speed     : regression chuẩn
  - cloud          : sigmoid 0-1 (độ mây)

Cách dùng:
    python train_tcn_hard.py --csv data.csv --horizon 6 --lookback 48 --epochs 150

Tùy chỉnh:
    --csv           : Đường dẫn file CSV
    --horizon       : Số bước dự báo (mặc định: 6)
    --lookback      : Số bước đầu vào (mặc định: 48)
    --epochs        : Số epoch (mặc định: 150)
    --batch_size    : Batch size (mặc định: 64)
    --lr            : Learning rate (mặc định: 0.0005)
    --num_channels  : Kênh ẩn mỗi lớp TCN (mặc định: 128 128 64 64 32)
    --kernel_size   : Kernel size (mặc định: 7)
    --dropout       : Dropout (mặc định: 0.2)
    --patience      : Early stopping patience (mặc định: 20)
    --output        : File model đầu ra (mặc định: tcn_hard_targets.pth)
=============================================================================
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import torch.nn.functional as F

# =============================================================================
# 1. TCN MODEL (giống file gốc)
# =============================================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)


class TCNBackbone(nn.Module):
    """TCN backbone dùng chung cho các head khác nhau."""
    def __init__(self, num_features, num_channels, kernel_size, dropout):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            in_ch = num_features if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, lookback, features) -> (batch, features, lookback)
        x = x.transpose(1, 2)
        out = self.tcn(x)       # (batch, channels, lookback)
        out = out[:, :, -1]     # (batch, channels) - timestep cuối
        return out


class HardTargetForecaster(nn.Module):
    """
    Multi-head TCN cho các target khó.

    Các head:
      - wind_dir_head  : dự báo sin/cos của hướng gió → 2 * horizon outputs
      - precip_head    : regression lượng mưa (mm, scaled) → horizon outputs
      - rain_prob_head : regression chuẩn → horizon outputs
      - gust_head      : regression chuẩn → horizon outputs
      - cloud_head     : regression 0-100 → horizon outputs
    """
    def __init__(self, num_features, horizon, num_channels, kernel_size, dropout):
        super().__init__()
        self.horizon = horizon
        last_ch = num_channels[-1]

        self.backbone = TCNBackbone(num_features, num_channels, kernel_size, dropout)

        # Head cho wind_direction: dự báo sin và cos
        self.wind_dir_head = nn.Sequential(
            nn.Linear(last_ch, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon * 2)  # sin + cos cho mỗi bước
        )

        # Head cho precipitation: regression (lượng mưa mm, scaled)
        self.precip_head = nn.Sequential(
            nn.Linear(last_ch, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon),
        )

        # Head cho rain_probability: regression 0-100
        self.rain_prob_head = nn.Sequential(
            nn.Linear(last_ch, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon),
            nn.Sigmoid()  # output 0-1, nhân 100 khi inference
        )

        # Head cho gust_speed: regression chuẩn
        # self.gust_head = nn.Sequential(
        #     nn.Linear(last_ch, 64),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(64, horizon)
        # )

        # Head cho cloud: regression 0-100
        self.cloud_head = nn.Sequential(
            nn.Linear(last_ch, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon),
            nn.Sigmoid()  # output 0-1, nhân 100 khi inference
        )

    def forward(self, x):
        feat = self.backbone(x)  # (batch, last_ch)

        #wind_dir = self.wind_dir_head(feat).view(-1, self.horizon, 2)  # (B, H, 2) sin+cos
        wind_dir_raw = self.wind_dir_head(feat).view(-1, self.horizon, 2)
        wind_dir = torch.nn.functional.normalize(wind_dir_raw, p=2, dim=-1)  # ép lên unit circle
        precip = self.precip_head(feat).view(-1, self.horizon, 1)
        rain_prob = self.rain_prob_head(feat).view(-1, self.horizon, 1)  # 0-1
        # gust = self.gust_head(feat).view(-1, self.horizon, 1)
        cloud = self.cloud_head(feat).view(-1, self.horizon, 1)  # 0-1

        return {
            'wind_dir': wind_dir,           # (B, H, 2) sin, cos
            'precip': precip,               # (B, H, 1) scaled mm
            'rain_prob': rain_prob,         # (B, H, 1) 0-1
            # 'gust': gust,                   # (B, H, 1) scaled
            'cloud': cloud,                 # (B, H, 1) 0-1
        }


# =============================================================================
# 2. CUSTOM LOSS
# =============================================================================

class HardTargetLoss(nn.Module):
    """
    Loss function tùy chỉnh cho từng target:
      - wind_dir    : MSE trên sin/cos (tránh vấn đề góc tròn)
      - precip      : MSE (lượng mưa)
      - rain_prob   : BCE loss (0-1)
      - gust        : MSE
      - cloud       : BCE loss (0-1)
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        losses = {}

        # Wind direction: MSE trên sin/cos
        losses['wind_dir'] = self.mse(pred['wind_dir'], target['wind_dir'])

        # Precipitation: MSE
        losses['precip'] = self.mse(pred['precip'], target['precip'])

        # Rain probability: BCE
        #losses['rain_prob'] = self.bce(pred['rain_prob'], target['rain_prob'])
        losses['rain_prob'] = weighted_bce(pred['rain_prob'], target['rain_prob'])

        # Gust speed: MSE
        # losses['gust'] = self.mse(pred['gust'], target['gust'])
        
        # Cloud: BCE
        losses['cloud'] = self.bce(pred['cloud'], target['cloud'])
        # losses['cloud'] = self.bce(pred['cloud'], target['cloud']) \
        #         + 0.3 * variance_penalty(pred['cloud'])

        # Weighted total
        total = (
            2.0 * losses['wind_dir'] +
            1.5 * losses['precip'] +
            2.0 * losses['rain_prob'] +
            1.5 * losses['cloud']
        )
        # nếu sài thêm cái này vô 1.0 * losses['gust'] +
        return total, losses


# =============================================================================
# 3. DATASET
# =============================================================================

class HardTargetDataset(Dataset):
    """
    Dataset chuyên cho các target khó.
    Tự chuyển đổi wind_direction → sin/cos, v.v.
    """
    def __init__(self, features, targets_dict, lookback, horizon):
        """
        targets_dict chứa:
          'wind_dir_sin', 'wind_dir_cos': (N,) mỗi cái
          'precip': (N,) lượng mưa scaled
          'rain_prob': (N,) 0-1
          'gust': (N,) scaled
          'cloud': (N,) 0-1
        """
        self.features = features
        self.targets = targets_dict
        self.lookback = lookback
        self.horizon = horizon
        self.length = len(features) - lookback - horizon + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.lookback]
        s = idx + self.lookback
        e = s + self.horizon

        y = {
            'wind_dir': np.stack([
                self.targets['wind_dir_sin'][s:e],
                self.targets['wind_dir_cos'][s:e]
            ], axis=-1),  # (H, 2)
            'precip': self.targets['precip'][s:e, None],  # (H, 1)
            'rain_prob': self.targets['rain_prob'][s:e, None],  # (H, 1)
            # 'gust': self.targets['gust'][s:e, None],  # (H, 1)
            'cloud': self.targets['cloud'][s:e, None],  # (H, 1)
        }

        x_t = torch.FloatTensor(x)
        y_t = {k: torch.FloatTensor(v) for k, v in y.items()}
        return x_t, y_t



def circular_interpolate(series_deg):
    """Interpolate góc tròn qua sin/cos để tránh lỗi 350→180→10."""
    rad = np.radians(series_deg)
    s = pd.Series(np.sin(rad), index=series_deg.index)
    c = pd.Series(np.cos(rad), index=series_deg.index)
    s = s.interpolate(method='linear', limit_direction='both')
    c = c.interpolate(method='linear', limit_direction='both')
    return np.degrees(np.arctan2(s, c)) % 360

def variance_penalty(pred_tensor, min_std=15.0):
    """Phạt nếu std của prediction < min_std (quá smooth)."""
    pred_std = pred_tensor.std(dim=0).mean()
    penalty = F.relu(min_std / 100.0 - pred_std)  # /100 vì cloud là 0-1
    return penalty

def weighted_bce(pred, target, pos_weight=2.36):
    weight = torch.where(
        target > 0.5,
        torch.full_like(target, pos_weight),
        torch.ones_like(target)
    )
    return F.binary_cross_entropy(pred, target, weight=weight)


# =============================================================================
# 4. DATA PROCESSING
# =============================================================================

def load_and_preprocess(csv_path):
    """Đọc CSV và tạo features + targets đặc biệt cho hard targets."""

    """đọc file csv và tạo featture + target đặc biệt cho hard target
    Tiền xử lý đầy đủ:
      1. Loại bỏ dòng trùng lặp
      2. Lấp khoảng trống thời gian (reindex hourly)
      3. Clip giá trị ngoại lai theo domain knowledge
      4. Thêm đặc trưng thời gian cyclical
      5. Interpolate giá trị thiếu
    """
    df = pd.read_csv(csv_path)
    print(f"[INFO] Read {len(df)} rows from {csv_path}")

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)


    # --- bước 1 loại bỏ trùng lặp ---
    dup_full = df.duplicated().sum()
    df = df.drop_duplicates()
    dup_ts = df['timestamp'].duplicated().sum()
    df = df.drop_duplicates(subset='timestamp', keep='last')
    print(f"[PREPROCESS] remove {dup_full} completely duplicated lines, {dup_ts} duplicated timestamp lines")
    

    # --- bước 2 lấp đầy khoảng trống thời gian ---
    
    full_range = pd.date_range(df['timestamp'].min(), df['timestamp'].max(), freq='h')
    missing_hours = len(full_range) - len(df)
    df = df.set_index('timestamp').reindex(full_range)
    df.index.name = 'timestamp'
    df = df.reset_index()
    print(f"[PREPROCESS] fill {missing_hours} hours missing (reindex hourly)")



    # --- bước 3 clip giá trị ngoại lai theo domain knowledge ---
    BOUNDS = {
        'temperature':   (15, 45),
        'feels_like':    (15, 55),
        'humidity':      (5, 100),
        'wind_speed':    (0, 60),
        'gust_speed':    (0, 80),
        'pressure':      (990, 1030),
        'precipitation': (0, 100),
        'dewpoint':      (5, 35),
        'visibility':    (0, 50),
        'cloud':         (0, 100),
        'uv_index':      (0, 15),
        'wind_direction': (0, 360),
        'rain_probability': (0, 100),
    }
    clipped = 0
    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            outliers = ((df[col] < lo) | (df[col] > hi)).sum()
            clipped += outliers
            df[col] = df[col].clip(lo, hi)
    print(f"[PREPROCESS] Clip {clipped} outlier value")
    
    # --- bước 4: đặc trưng thời gian cyclical ----
    df['hour_sin']  = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.month / 12)
    df['dow_sin']   = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
    df['dow_cos']   = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
    print(f"[PREPROCESS] added 6 feature  cyclical (hour/month/dow sin+cos)")

    # --- Feature columns (tất cả cột số, trừ các cột không liên quan) ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['latitude', 'longitude', 'condition_code']
    # feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    feature_cols_base = [c for c in numeric_cols if c not in exclude_cols]


    # Interpolate tất cả feature gốc (giữ wind_direction tạm thời)
    cols_to_interp = [c for c in feature_cols_base if c != 'wind_direction']
    df[cols_to_interp] = df[cols_to_interp].interpolate(method='linear', limit_direction='both')
    df[cols_to_interp] = df[cols_to_interp].ffill().bfill()

    # Interpolate target phụ
    for col in ['precipitation', 'rain_probability', 'cloud']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit_direction='both').ffill().bfill()
    # wind_direction circular interpolate
    if 'wind_direction' in df.columns:
        df['wind_direction'] = circular_interpolate(df['wind_direction'])
    # Tạo sin/cos SAU khi wind_direction đã đúng
    df['wind_dir_sin'] = np.sin(np.radians(df['wind_direction']))
    df['wind_dir_cos'] = np.cos(np.radians(df['wind_direction']))

    # --- Thêm sin/cos wind_direction vào features ---
    # df['wind_dir_sin'] = np.sin(np.radians(df['wind_direction']))
    # df['wind_dir_cos'] = np.cos(np.radians(df['wind_direction']))

    # Thay wind_direction bằng sin/cos trong features
    feature_cols = [c if c != 'wind_direction' else 'wind_dir_sin' for c in feature_cols_base]

    if 'wind_dir_sin' in feature_cols:
        idx = feature_cols.index('wind_dir_sin')
        feature_cols.insert(idx + 1, 'wind_dir_cos')
    # Loại bỏ duplicates
    feature_cols = list(dict.fromkeys(feature_cols))


    # ===== BƯỚC 5: Interpolate giá trị thiếu (sau reindex) =====
    # Xử lý giá trị thiếu
    df[feature_cols] = df[feature_cols].interpolate(method='linear', limit_direction='both')
    df[feature_cols] = df[feature_cols].ffill().bfill()

    print(f"[INFO] Feature columns ({len(feature_cols)}): {feature_cols}")


    # Fill cột chuỗi
    str_cols = df.select_dtypes(include=['object']).columns.tolist()
    if str_cols:
        df[str_cols] = df[str_cols].ffill().bfill()
 
    remaining_na = df[feature_cols].isna().sum().sum()
    print(f"[PREPROCESS] Interpolate xong, còn {remaining_na} NaN")
    print(f"[INFO] Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"[INFO] Final shape: {df.shape}")
 
    # Interpolate các cột nguồn target (có thể NaN sau reindex)
    #target_source_cols = ['wind_direction', 'precipitation', 'rain_probability', 'cloud'] #, 'gust_speed' nếu dùng
    # for col in target_source_cols:
    #     if col in df.columns:
    #         df[col] = df[col].interpolate(method='linear', limit_direction='both').ffill().bfill()

    # if 'wind_direction' in df.columns:
    #     df['wind_direction'] = circular_interpolate(df['wind_direction'])
    # Interpolate các target khác (không phải wind_direction)

    # --- Tạo target arrays ---
    # wind_direction → sin/cos
    wind_dir_sin = np.sin(np.radians(df['wind_direction'].values)).astype(np.float32)
    wind_dir_cos = np.cos(np.radians(df['wind_direction'].values)).astype(np.float32)

    # precipitation → amount (sẽ scale sau)
    precip = df['precipitation'].values.astype(np.float32)

    # rain_probability → 0-1
    rain_prob = (df['rain_probability'].values / 100.0).astype(np.float32)

    # gust_speed (sẽ scale sau)
    #gust = df['gust_speed'].values.astype(np.float32)

    # cloud → 0-1
    cloud = (df['cloud'].values / 100.0).astype(np.float32)

    targets_raw = {
        'wind_dir_sin': wind_dir_sin,
        'wind_dir_cos': wind_dir_cos,
        'precip': precip,
        'rain_prob': rain_prob,
        # 'gust': gust,
        'cloud': cloud,
    }

    return df, feature_cols, targets_raw


def create_datasets(df, feature_cols, targets_raw, lookback, horizon,
                    train_ratio, val_ratio):
    """Chia train/val/test, scale features, gust_speed và precipitation."""
    features = df[feature_cols].values.astype(np.float32)
    n = len(features)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Scale features
    feature_scaler = StandardScaler()
    feature_scaler.fit(features[:train_end])
    features_scaled = feature_scaler.transform(features)

    # Scale gust_speed riêng
    # gust_scaler = StandardScaler()
    # gust_raw = targets_raw['gust'].reshape(-1, 1)
    # gust_scaler.fit(gust_raw[:train_end])
    # gust_scaled = gust_scaler.transform(gust_raw).flatten()

    # Scale precipitation riêng
    precip_scaler = StandardScaler()
    precip_raw = targets_raw['precip'].reshape(-1, 1)
    precip_scaler.fit(precip_raw[:train_end])
    precip_scaled = precip_scaler.transform(precip_raw).flatten()

    # Tạo targets dict đã scale
    def make_targets(start, end):
        return {
            'wind_dir_sin': targets_raw['wind_dir_sin'][start:end],
            'wind_dir_cos': targets_raw['wind_dir_cos'][start:end],
            'precip': precip_scaled[start:end],
            'rain_prob': targets_raw['rain_prob'][start:end],  # đã 0-1
            # 'gust': gust_scaled[start:end],
            'cloud': targets_raw['cloud'][start:end],  # đã 0-1
        }

    train_ds = HardTargetDataset(features_scaled[:train_end], make_targets(0, train_end), lookback, horizon)
    val_ds = HardTargetDataset(features_scaled[train_end:val_end], make_targets(train_end, val_end), lookback, horizon)
    test_ds = HardTargetDataset(features_scaled[val_end:], make_targets(val_end, n), lookback, horizon)

    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    scalers = {
        'feature_scaler': feature_scaler,
        # 'gust_scaler': gust_scaler,
        'precip_scaler': precip_scaler,
    }

    return train_ds, val_ds, test_ds, scalers


# =============================================================================
# 5. TRAINING
# =============================================================================

def collate_fn(batch):
    """Custom collate để gom dict targets."""
    xs, ys = zip(*batch)
    x = torch.stack(xs)
    y = {}
    for key in ys[0].keys():
        y[key] = torch.stack([yi[key] for yi in ys])
    return x, y


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x = x.to(device)
        y = {k: v.to(device) for k, v in y.items()}
        pred = model(x)
        loss, _ = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_losses = {}
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = {k: v.to(device) for k, v in y.items()}
        pred = model(x)
        loss, sub_losses = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        count += x.size(0)
        for k, v in sub_losses.items():
            all_losses[k] = all_losses.get(k, 0) + v.item() * x.size(0)

    avg_losses = {k: v / count for k, v in all_losses.items()}
    return total_loss / count, avg_losses


@torch.no_grad()
def compute_metrics(model, loader, scalers, device, horizon):
    """Tính metrics trên giá trị gốc (inverse scaled)."""
    model.eval()
    all_preds = {k: [] for k in ['wind_dir', 'precip', 'rain_prob', 'cloud']}
    all_trues = {k: [] for k in ['wind_dir', 'precip', 'rain_prob', 'cloud']} #, 'gust' nếu dùng

    for x, y in loader:
        x = x.to(device)
        pred = model(x)
        for k in all_preds:
            all_preds[k].append(pred[k].cpu().numpy())
            all_trues[k].append(y[k].numpy())

    for k in all_preds:
        all_preds[k] = np.concatenate(all_preds[k], axis=0)
        all_trues[k] = np.concatenate(all_trues[k], axis=0)

    results = {}

    # --- Wind Direction: sin/cos → degrees ---
    for hi in range(horizon):
        pred_sin = all_preds['wind_dir'][:, hi, 0]
        pred_cos = all_preds['wind_dir'][:, hi, 1]
        true_sin = all_trues['wind_dir'][:, hi, 0]
        true_cos = all_trues['wind_dir'][:, hi, 1]

        pred_deg = np.degrees(np.arctan2(pred_sin, pred_cos)) % 360
        true_deg = np.degrees(np.arctan2(true_sin, true_cos)) % 360

        # Circular MAE: khoảng cách góc ngắn nhất
        diff = np.abs(pred_deg - true_deg)
        circular_diff = np.minimum(diff, 360 - diff)
        mae = np.mean(circular_diff)
        rmse = np.sqrt(np.mean(circular_diff ** 2))

        results[f"wind_direction_h{hi+1}"] = {"MAE(°)": float(mae), "RMSE(°)": float(rmse)}

    # --- Precipitation: MAE trên giá trị gốc ---
    precip_scaler = scalers['precip_scaler']
    for hi in range(horizon):
        pred_p = precip_scaler.inverse_transform(all_preds['precip'][:, hi].reshape(-1, 1)).flatten()
        true_p = precip_scaler.inverse_transform(all_trues['precip'][:, hi].reshape(-1, 1)).flatten()
        pred_p = np.clip(pred_p, 0, None)
        true_p = np.clip(true_p, 0, None)
        mae = np.mean(np.abs(pred_p - true_p))
        rmse = np.sqrt(np.mean((pred_p - true_p) ** 2))
        results[f"precipitation_h{hi+1}"] = {"MAE(mm)": float(mae), "RMSE(mm)": float(rmse)}

    # --- Rain Probability: MAE trên 0-100 ---
    for hi in range(horizon):
        pred_rp = all_preds['rain_prob'][:, hi, 0] * 100
        true_rp = all_trues['rain_prob'][:, hi, 0] * 100
        mae = np.mean(np.abs(pred_rp - true_rp))
        rmse = np.sqrt(np.mean((pred_rp - true_rp) ** 2))
        results[f"rain_probability_h{hi+1}"] = {"MAE(%)": float(mae), "RMSE(%)": float(rmse)}

    # --- Gust Speed: MAE trên giá trị gốc ---
    # gust_scaler = scalers['gust_scaler']
    # for hi in range(horizon):
    #     pred_g = gust_scaler.inverse_transform(all_preds['gust'][:, hi].reshape(-1, 1)).flatten()
    #     true_g = gust_scaler.inverse_transform(all_trues['gust'][:, hi].reshape(-1, 1)).flatten()
    #     mae = np.mean(np.abs(pred_g - true_g))
    #     rmse = np.sqrt(np.mean((pred_g - true_g) ** 2))
    #     results[f"gust_speed_h{hi+1}"] = {"MAE(km/h)": float(mae), "RMSE(km/h)": float(rmse)}

    # --- Cloud: MAE trên 0-100 ---
    for hi in range(horizon):
        pred_c = all_preds['cloud'][:, hi, 0] * 100
        true_c = all_trues['cloud'][:, hi, 0] * 100
        mae = np.mean(np.abs(pred_c - true_c))
        rmse = np.sqrt(np.mean((pred_c - true_c) ** 2))
        results[f"cloud_h{hi+1}"] = {"MAE(%)": float(mae), "RMSE(%)": float(rmse)}

    return results


# =============================================================================
# 6. PLOTTING
# =============================================================================

def plot_training_history(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Hard Targets - Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Chart saved: {save_path}")


@torch.no_grad()
def collect_predictions(model, loader, scalers, device, horizon):
    """Thu thập toàn bộ predictions và true values, chuyển về giá trị gốc."""
    model.eval()
    all_preds = {k: [] for k in ['wind_dir', 'precip', 'rain_prob', 'cloud']}
    all_trues = {k: [] for k in ['wind_dir', 'precip', 'rain_prob', 'cloud']} #, 'gust' nếu dùng

    for x, y in loader:
        x = x.to(device)
        pred = model(x)
        for k in all_preds:
            all_preds[k].append(pred[k].cpu().numpy())
            all_trues[k].append(y[k].numpy())

    for k in all_preds:
        all_preds[k] = np.concatenate(all_preds[k], axis=0)
        all_trues[k] = np.concatenate(all_trues[k], axis=0)

    # gust_scaler = scalers['gust_scaler']
    precip_scaler = scalers['precip_scaler']

    # Chuyển về giá trị gốc
    results = {}

    # 1. Wind direction: sin/cos → degrees
    last_h = horizon - 1
    pred_deg = np.degrees(np.arctan2(
        all_preds['wind_dir'][:, last_h, 0],
        all_preds['wind_dir'][:, last_h, 1]
    )) % 360
    true_deg = np.degrees(np.arctan2(
        all_trues['wind_dir'][:, last_h, 0],
        all_trues['wind_dir'][:, last_h, 1]
    )) % 360
    results['wind_direction'] = {'pred': pred_deg, 'true': true_deg, 'unit': '°'}

    # 2. Precipitation: inverse scale
    pred_p = precip_scaler.inverse_transform(
        all_preds['precip'][:, last_h].reshape(-1, 1)).flatten()
    true_p = precip_scaler.inverse_transform(
        all_trues['precip'][:, last_h].reshape(-1, 1)).flatten()
    pred_p = np.clip(pred_p, 0, None)
    true_p = np.clip(true_p, 0, None)
    results['precipitation'] = {'pred': pred_p, 'true': true_p, 'unit': 'mm'}

    # 3. Rain probability: ×100
    pred_rp = all_preds['rain_prob'][:, last_h, 0] * 100
    true_rp = all_trues['rain_prob'][:, last_h, 0] * 100
    results['rain_probability'] = {'pred': pred_rp, 'true': true_rp, 'unit': '%'}

    # 4. Gust speed: inverse scale
    # pred_g = gust_scaler.inverse_transform(
    #     all_preds['gust'][:, last_h].reshape(-1, 1)).flatten()
    # true_g = gust_scaler.inverse_transform(
    #     all_trues['gust'][:, last_h].reshape(-1, 1)).flatten()
    # results['gust_speed'] = {'pred': pred_g, 'true': true_g, 'unit': 'km/h'}

    # 5. Cloud: ×100
    pred_c = all_preds['cloud'][:, last_h, 0] * 100
    true_c = all_trues['cloud'][:, last_h, 0] * 100
    results['cloud'] = {'pred': pred_c, 'true': true_c, 'unit': '%'}

    return results


def plot_predictions(prediction_data, horizon, save_path, num_samples=200):
    """Vẽ biểu đồ so sánh dự báo vs thực tế cho từng target."""
    targets = list(prediction_data.keys())
    num_targets = len(targets)

    fig, axes = plt.subplots(num_targets, 1, figsize=(14, 4 * num_targets), sharex=True)
    if num_targets == 1:
        axes = [axes]

    for i, target_name in enumerate(targets):
        data = prediction_data[target_name]
        n = min(num_samples, len(data['pred']))
        pred = data['pred'][:n]
        true = data['true'][:n]
        unit = data['unit']

        axes[i].plot(true, label='reality', linewidth=1.5, alpha=0.8, color='#1f77b4')
        axes[i].plot(pred, label=f'forecast (h+{horizon})', linewidth=1.5, alpha=0.8, color='#ff7f0e')
        axes[i].set_ylabel(f"{target_name} ({unit})")
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)

        # Hiển thị MAE trên biểu đồ
        if target_name == 'wind_direction':
            diff = np.abs(pred - true)
            circular_diff = np.minimum(diff, 360 - diff)
            mae = np.mean(circular_diff)
        else:
            mae = np.mean(np.abs(pred - true))
        axes[i].set_title(f"{target_name} — MAE: {mae:.2f} {unit}", fontsize=10, loc='left')

    axes[-1].set_xlabel('Sample index')
    fig.suptitle(f'Hard Targets: forecast vs reality (horizon = {horizon}h)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Chart saved predictions: {save_path}")


def plot_sub_losses(sub_losses_history, save_path):
    """Vẽ biểu đồ sub-losses theo epoch."""
    plt.figure(figsize=(12, 6))
    for key in sub_losses_history[0].keys():
        values = [sl[key] for sl in sub_losses_history]
        plt.plot(values, label=key, linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Sub-Losses per Target')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Chart saved: {save_path}")


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train TCN for hard weather targets")

    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--lookback', type=int, default=48)
    parser.add_argument('--horizon', type=int, default=6)
    parser.add_argument('--num_channels', nargs='+', type=int, default=[128, 128, 64, 64, 32])
    parser.add_argument('--kernel_size', type=int, default=7)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--output', type=str, default='tcn_hard_targets.pth')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Lookback: {args.lookback}h | Horizon: {args.horizon}h")
    print(f"[INFO] Hard targets: wind_direction, precipitation, rain_probability, cloud (no precip_has_rain/precip_amount)")
    print("=" * 70)

    # --- Load data ---
    df, feature_cols, targets_raw = load_and_preprocess(args.csv)

    # --- Create datasets ---
    train_ds, val_ds, test_ds, scalers = create_datasets(
        df, feature_cols, targets_raw,
        args.lookback, args.horizon,
        args.train_ratio, args.val_ratio
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn)

    # --- Build model ---
    num_features = len(feature_cols)

    model = HardTargetForecaster(
        num_features=num_features,
        horizon=args.horizon,
        num_channels=args.num_channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model params: {total_params:,}")

    # --- Training ---
    criterion = HardTargetLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_sub_losses_history = []

    print("\n" + "=" * 70)
    print("START TRAINING")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_sub = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_sub_losses_history.append(val_sub)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), '_best_hard.pth')
            marker = ' *** best ***'
        else:
            patience_counter += 1
            marker = ''

        if epoch % 5 == 0 or epoch == 1 or marker:
            sub_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_sub.items()])
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f}{marker}")
            if epoch % 10 == 0:
                print(f"    Sub-losses: {sub_str}")

        if patience_counter >= args.patience:
            print(f"\n[INFO] Early stopping at epoch {epoch}")
            break

    # Load best
    model.load_state_dict(torch.load('_best_hard.pth', weights_only=True))
    os.remove('_best_hard.pth')

    # --- Evaluate ---
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)

    test_loss, test_sub = evaluate(model, test_loader, criterion, device)
    print(f"  Total Test Loss: {test_loss:.6f}")
    for k, v in test_sub.items():
        print(f"    {k}: {v:.6f}")

    metrics = compute_metrics(model, test_loader, scalers, device, args.horizon)

    print(f"\n  {'Metric':<35s} {'Value 1':>12s} {'Value 2':>12s}")
    print("  " + "-" * 60)
    for key, vals in metrics.items():
        val_strs = [f"{v:>12.4f}" for v in vals.values()]
        labels = list(vals.keys())
        print(f"  {key:<35s} {val_strs[0]} ({labels[0]})")
        if len(val_strs) > 1:
            print(f"  {'':<35s} {val_strs[1]} ({labels[1]})")

    # --- Save ---
    output_dir = os.path.dirname(args.output) or '.'
    os.makedirs(output_dir, exist_ok=True)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_features': num_features,
            'horizon': args.horizon,
            'lookback': args.lookback,
            'num_channels': args.num_channels,
            'kernel_size': args.kernel_size,
            'dropout': args.dropout,
        },
        'feature_cols': feature_cols,
        'hard_targets': ['wind_direction', 'precipitation', 'rain_probability', 'cloud'],
        'feature_scaler_mean': scalers['feature_scaler'].mean_.tolist(),
        'feature_scaler_scale': scalers['feature_scaler'].scale_.tolist(),
        # 'gust_scaler_mean': float(scalers['gust_scaler'].mean_[0]),
        # 'gust_scaler_scale': float(scalers['gust_scaler'].scale_[0]),
        'precip_scaler_mean': float(scalers['precip_scaler'].mean_[0]),
        'precip_scaler_scale': float(scalers['precip_scaler'].scale_[0]),
        'test_metrics': metrics,
        'train_info': {
            'epochs_trained': len(train_losses),
            'best_val_loss': float(best_val_loss),
            'test_loss': float(test_loss),
            'total_params': total_params,
            'device': str(device),
            'timestamp': datetime.now().isoformat(),
        }
    }

    torch.save(save_dict, args.output)
    print(f"\n[INFO] Model saved: {args.output}")

    # --- Plots ---
    base_name = os.path.splitext(args.output)[0]
    plot_training_history(train_losses, val_losses, f"{base_name}_training_history.png")
    if val_sub_losses_history:
        plot_sub_losses(val_sub_losses_history, f"{base_name}_sub_losses.png")

    # Predictions plot
    prediction_data = collect_predictions(model, test_loader, scalers, device, args.horizon)
    plot_predictions(prediction_data, args.horizon, f"{base_name}_predictions.png")

    # --- Metrics JSON ---
    metrics_path = f"{base_name}_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': vars(args),
            'test_metrics': metrics,
            'train_info': save_dict['train_info']
        }, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Metrics saved: {metrics_path}")

    print("\n" + "=" * 70)
    print("COMPLETED!")
    print("=" * 70)


if __name__ == '__main__':
    main()