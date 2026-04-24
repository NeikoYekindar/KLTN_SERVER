"""
=============================================================================
Inference Script - Dual TCN (tcn_model_24h + tcn_hard_model_24h)
=============================================================================
Two models with different architectures:

  TCNForecaster        (tcn_model_24h.pth  - train_new.py)
      targets : temperature, humidity, wind_speed, pressure,
                uv_index, dewpoint, visibility
      scaler  : target_scaler_mean / target_scaler_scale  (StandardScaler)

  HardTargetForecaster (tcn_hard_model_24h.pth - train_tcn_hard.py)
      targets : wind_direction, precipitation, cloud
      scaler  : precip_scaler (separate)
      special : features need wind_dir_sin / wind_dir_cos
                wind_dir output = sin+cos -> atan2 -> degrees

  RainProbClassifier   (rain_prob_classifier.pkl)
      input   : humidity, cloud, pressure, temperature,
                precipitation, wind_speed, visibility
      output  : rain_probability — 0 / 45 / 100 (%)

  ConditionClassifier  (condition_classifier.pkl)
      input   : output của 2 TCN model + rain_probability
      output  : "Clear", "Cloudy", "Rain", ...

Usage:
    python inference_dual_tcn.py \
        --tcn      tcn_model_24h.pth \
        --tcn_hard tcn_hard_model_24h.pth \
        --csv      latest_48h.csv

    python inference_dual_tcn.py \
        --tcn                  tcn_model_24h.pth \
        --tcn_hard             tcn_hard_model_24h.pth \
        --csv                  latest_48h.csv \
        --output               forecast_24h.json \
        --device               cuda \
        --classifier           condition_classifier.pkl \
        --rain_prob_classifier rain_prob_classifier.pkl
=============================================================================
"""

import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. PyTorch / Model definitions
# ============================================================
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

if TORCH_OK:

    # ----------------------------------------------------------
    # Shared building blocks
    # ----------------------------------------------------------
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
            self.conv1      = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
            self.bn1        = nn.BatchNorm1d(out_channels)
            self.conv2      = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
            self.bn2        = nn.BatchNorm1d(out_channels)
            self.dropout    = nn.Dropout(dropout)
            self.relu       = nn.ReLU()
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

    # ----------------------------------------------------------
    # Model 1: TCNForecaster  (train_new.py)
    # ----------------------------------------------------------
    class TCNForecaster(nn.Module):
        def __init__(self, num_features, num_targets, horizon,
                     num_channels, kernel_size, dropout):
            super().__init__()
            self.horizon     = horizon
            self.num_targets = num_targets
            layers = []
            for i in range(len(num_channels)):
                in_ch = num_features if i == 0 else num_channels[i - 1]
                layers.append(
                    TemporalBlock(in_ch, num_channels[i], kernel_size, 2 ** i, dropout)
                )
            self.tcn = nn.Sequential(*layers)
            self.fc  = nn.Sequential(
                nn.Linear(num_channels[-1], 128), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, horizon * num_targets)
            )

        def forward(self, x):
            x   = x.transpose(1, 2)
            out = self.tcn(x)[:, :, -1]
            return self.fc(out).view(-1, self.horizon, self.num_targets)

    # ----------------------------------------------------------
    # Model 2: HardTargetForecaster  (train_tcn_hard.py)
    # rain_probability đã bị loại bỏ — chỉ còn wind_dir, precip, cloud
    # ----------------------------------------------------------
    class TCNBackbone(nn.Module):
        def __init__(self, num_features, num_channels, kernel_size, dropout):
            super().__init__()
            layers = []
            for i in range(len(num_channels)):
                in_ch = num_features if i == 0 else num_channels[i - 1]
                layers.append(
                    TemporalBlock(in_ch, num_channels[i], kernel_size, 2 ** i, dropout)
                )
            self.tcn = nn.Sequential(*layers)

        def forward(self, x):
            x   = x.transpose(1, 2)
            out = self.tcn(x)
            return out[:, :, -1]   # (batch, last_ch)

    class HardTargetForecaster(nn.Module):
        def __init__(self, num_features, horizon, num_channels, kernel_size, dropout):
            super().__init__()
            self.horizon = horizon
            last_ch = num_channels[-1]
            self.backbone = TCNBackbone(num_features, num_channels, kernel_size, dropout)

            def _head(out_size, sigmoid=False):
                layers = [
                    nn.Linear(last_ch, 64), nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, out_size),
                ]
                if sigmoid:
                    layers.append(nn.Sigmoid())
                return nn.Sequential(*layers)

            self.wind_dir_head = _head(horizon * 2)
            self.precip_head   = nn.Sequential(
                nn.Linear(last_ch, 64), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, horizon),
            )
            # rain_prob_head đã bị loại — dùng RainProbClassifier riêng
            self.cloud_head    = _head(horizon, sigmoid=True)

        def forward(self, x):
            feat = self.backbone(x)
            return {
                'wind_dir': self.wind_dir_head(feat).view(-1, self.horizon, 2),
                'precip':   self.precip_head(feat).view(-1, self.horizon, 1),
                'cloud':    self.cloud_head(feat).view(-1, self.horizon, 1),
            }


# ============================================================
# 2. Data helpers
# ============================================================

def load_csv(csv_path: str, feature_cols: list, lookback: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"  [DATA] Read {len(df)} rows from {csv_path}")

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    if len(df) < lookback:
        raise ValueError(
            f"[ERROR] CSV only has {len(df)} rows, need at least {lookback}."
        )

    df = df.tail(lookback).reset_index(drop=True)

    # Fill missing
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].interpolate(method='linear').ffill().bfill()

    # Compute cyclical features if needed
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        if 'hour_sin' in feature_cols and 'hour_sin' not in df.columns:
            df['hour_sin']  = np.sin(2 * np.pi * ts.dt.hour / 24)
            df['hour_cos']  = np.cos(2 * np.pi * ts.dt.hour / 24)
        if 'month_sin' in feature_cols and 'month_sin' not in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * ts.dt.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * ts.dt.month / 12)
        if 'dow_sin' in feature_cols and 'dow_sin' not in df.columns:
            df['dow_sin']   = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
            df['dow_cos']   = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing columns in CSV: {missing}")

    return df


def get_last_timestamp(df: pd.DataFrame) -> datetime:
    if 'timestamp' in df.columns:
        return pd.to_datetime(df['timestamp'].iloc[-1])
    return datetime.now()


# ============================================================
# 3. Inference - TCNForecaster
# ============================================================

def predict_tcn(tcn_path: str, csv_path: str, device_str: str = 'cpu'):
    if not TORCH_OK:
        raise ImportError("PyTorch required: pip install torch")

    device = torch.device(device_str)
    ckpt   = torch.load(tcn_path, map_location=device, weights_only=False)
    cfg    = ckpt['model_config']

    feature_cols = ckpt['feature_cols']
    target_cols  = ckpt['target_cols']
    lookback     = cfg['lookback']
    horizon      = cfg['horizon']
    num_features = cfg.get('num_features', len(feature_cols))
    num_targets  = cfg.get('num_targets',  len(target_cols))

    model = TCNForecaster(
        num_features = num_features,
        num_targets  = num_targets,
        horizon      = horizon,
        num_channels = cfg['num_channels'],
        kernel_size  = cfg['kernel_size'],
        dropout      = cfg['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    feat_sc = StandardScaler()
    feat_sc.mean_          = np.array(ckpt['feature_scaler_mean'])
    feat_sc.scale_         = np.array(ckpt['feature_scaler_scale'])
    feat_sc.var_           = feat_sc.scale_ ** 2
    feat_sc.n_features_in_ = len(feat_sc.mean_)

    tgt_sc = StandardScaler()
    tgt_sc.mean_           = np.array(ckpt['target_scaler_mean'])
    tgt_sc.scale_          = np.array(ckpt['target_scaler_scale'])
    tgt_sc.var_            = tgt_sc.scale_ ** 2
    tgt_sc.n_features_in_  = len(tgt_sc.mean_)

    df  = load_csv(csv_path, feature_cols, lookback)
    X   = feat_sc.transform(df[feature_cols].values.astype(np.float32))
    x_t = torch.FloatTensor(X).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x_t).cpu().numpy()

    pred = tgt_sc.inverse_transform(
        out[0].reshape(-1, num_targets)
    ).reshape(horizon, num_targets)

    last_ts = get_last_timestamp(df)
    return pred, target_cols, horizon, last_ts


# ============================================================
# 4. Inference - HardTargetForecaster
# rain_probability KHÔNG còn ở đây — đã chuyển sang RainProbClassifier
# ============================================================

def predict_tcn_hard(tcn_hard_path: str, csv_path: str, device_str: str = 'cpu'):
    if not TORCH_OK:
        raise ImportError("PyTorch required: pip install torch")

    device = torch.device(device_str)
    ckpt   = torch.load(tcn_hard_path, map_location=device, weights_only=False)
    cfg    = ckpt['model_config']

    feature_cols = ckpt['feature_cols']
    lookback     = cfg['lookback']
    horizon      = cfg['horizon']
    num_features = cfg.get('num_features', len(feature_cols))

    model = HardTargetForecaster(
        num_features = num_features,
        horizon      = horizon,
        num_channels = cfg['num_channels'],
        kernel_size  = cfg['kernel_size'],
        dropout      = cfg['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    feat_sc = StandardScaler()
    feat_sc.mean_          = np.array(ckpt['feature_scaler_mean'])
    feat_sc.scale_         = np.array(ckpt['feature_scaler_scale'])
    feat_sc.var_           = feat_sc.scale_ ** 2
    feat_sc.n_features_in_ = len(feat_sc.mean_)

    precip_sc = StandardScaler()
    precip_sc.mean_          = np.array([ckpt['precip_scaler_mean']])
    precip_sc.scale_         = np.array([ckpt['precip_scaler_scale']])
    precip_sc.var_           = precip_sc.scale_ ** 2
    precip_sc.n_features_in_ = 1

    raw_cols_needed = [
        c for c in feature_cols
        if c not in ('wind_dir_sin', 'wind_dir_cos')
    ]
    base_need = raw_cols_needed + (
        ['wind_direction']
        if ('wind_dir_sin' in feature_cols and 'wind_direction' not in raw_cols_needed)
        else []
    )

    df = load_csv(csv_path, base_need, lookback)

    if 'wind_dir_sin' in feature_cols and 'wind_dir_sin' not in df.columns:
        df['wind_dir_sin'] = np.sin(np.radians(df['wind_direction']))
    if 'wind_dir_cos' in feature_cols and 'wind_dir_cos' not in df.columns:
        df['wind_dir_cos'] = np.cos(np.radians(df['wind_direction']))

    X   = feat_sc.transform(df[feature_cols].values.astype(np.float32))
    x_t = torch.FloatTensor(X).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x_t)

    # wind_direction: sin/cos → degrees
    wind_sin = out['wind_dir'][0, :, 0].cpu().numpy()
    wind_cos = out['wind_dir'][0, :, 1].cpu().numpy()
    wind_deg = (np.degrees(np.arctan2(wind_sin, wind_cos)) % 360)

    # precipitation: inverse scale + clip
    precip_sc_in = out['precip'][0, :, 0].cpu().numpy().reshape(-1, 1)
    precip_mm    = precip_sc.inverse_transform(precip_sc_in).flatten()
    precip_mm    = np.clip(precip_mm, 0.0, None)

    # cloud: sigmoid output × 100
    cloud_pct = out['cloud'][0, :, 0].cpu().numpy() * 100.0

    # rain_probability KHÔNG còn ở đây
    pred = np.stack([wind_deg, precip_mm, cloud_pct], axis=1)
    target_cols = ['wind_direction', 'precipitation', 'cloud']

    last_ts = get_last_timestamp(df)
    return pred, target_cols, horizon, last_ts


# ============================================================
# 5. Post-processing
# ============================================================

def postprocess(pred: np.ndarray, target_cols: list) -> np.ndarray:
    pred = pred.copy()
    for ti, col in enumerate(target_cols):
        c = col.lower()
        if any(k in c for k in ('humidity', 'cloud')):
            pred[:, ti] = np.clip(pred[:, ti], 0.0, 100.0)
        elif any(k in c for k in ('wind_speed', 'uv_index',
                                   'visibility', 'precipitation')):
            pred[:, ti] = np.clip(pred[:, ti], 0.0, None)
        elif 'wind_direction' in c:
            pred[:, ti] = pred[:, ti] % 360.0
    return pred


# ============================================================
# 5b. Rain Probability Classifier
# ============================================================

def load_rain_prob_classifier(classifier_path: str) -> dict:
    """Load rain probability classifier từ file .pkl"""
    import pickle
    with open(classifier_path, 'rb') as f:
        ckpt = pickle.load(f)
    print(f"  [RAIN_PROB] Loaded: {classifier_path}")
    print(f"  [RAIN_PROB] Classes: {ckpt['class_names']}")
    return ckpt


def predict_rain_probabilities(rain_prob_ckpt: dict, forecast: list) -> list:
    """
    Dự đoán rain_probability (0 / 45 / 100) cho từng step.

    Args:
        rain_prob_ckpt : dict từ load_rain_prob_classifier()
        forecast       : list of dict, mỗi dict là 1 step forecast
                         (cần có đủ các feature trong ckpt['features'])

    Returns:
        list of float: 0.0, 45.0, hoặc 100.0 cho mỗi step
    """
    model          = rain_prob_ckpt['model']
    scaler         = rain_prob_ckpt['scaler']
    features       = rain_prob_ckpt['features']
    class_to_value = rain_prob_ckpt['class_to_value']

    rain_probs = []
    for step in forecast:
        try:
            x = np.array(
                [[step.get(f, 0.0) for f in features]],
                dtype=np.float32
            )
            x_scaled  = scaler.transform(x)
            cls_idx   = int(model.predict(x_scaled)[0])
            rain_probs.append(float(class_to_value[cls_idx]))
        except Exception as e:
            print(f"  [RAIN_PROB WARN] Step {step.get('horizon_step')}: {e}")
            rain_probs.append(0.0)

    return rain_probs


# ============================================================
# 5c. Condition Classifier
# ============================================================

def load_condition_classifier(classifier_path: str) -> dict:
    """Load condition classifier từ file .pkl"""
    import pickle
    with open(classifier_path, 'rb') as f:
        ckpt = pickle.load(f)
    print(f"  [CLASSIFIER] Loaded: {classifier_path}")
    print(f"  [CLASSIFIER] Classes ({len(ckpt['classes'])}): {ckpt['classes']}")
    return ckpt


def predict_conditions(classifier_ckpt: dict, forecast: list) -> list:
    """Dự đoán condition cho từng step trong forecast."""
    model    = classifier_ckpt['model']
    scaler   = classifier_ckpt['scaler']
    le       = classifier_ckpt['label_encoder']
    features = classifier_ckpt['features']

    conditions = []
    for step in forecast:
        try:
            x        = np.array([[step.get(f, 0.0) for f in features]], dtype=np.float32)
            x_scaled = scaler.transform(x)
            pred_idx = model.predict(x_scaled)[0]
            conditions.append(le.inverse_transform([pred_idx])[0])
        except Exception as e:
            print(f"  [CLASSIFIER WARN] Step {step.get('horizon_step')}: {e}")
            conditions.append("Unknown")

    return conditions


# ============================================================
# 6. Output
# ============================================================

UNITS = {
    'temperature':      'C',
    'humidity':         '%',
    'wind_speed':       'km/h',
    'wind_direction':   'deg',
    'pressure':         'hPa',
    'precipitation':    'mm',
    'rain_probability': '%',
    'visibility':       'km',
    'cloud':            '%',
    'condition':        '',
}


def print_table(pred, cols, horizon, last_ts, title=''):
    print('\n' + '=' * 76)
    print(f'  {title}  -  next {horizon}h')
    print(f'  Data until: {last_ts.strftime("%Y-%m-%d %H:%M")}')
    print('=' * 76)
    header = f"  {'Timestamp':<20s}"
    for c in cols:
        u = UNITS.get(c, '')
        header += f"  {(c+'('+u+')' if u else c):>17s}"
    print(header)
    print('  ' + '-' * (18 + 19 * len(cols)))
    for hi in range(horizon):
        ts  = (last_ts + timedelta(hours=hi + 1)).strftime('%Y-%m-%d %H:%M')
        row = f"  {ts:<20s}"
        for ti in range(len(cols)):
            row += f"  {pred[hi, ti]:>17.3f}"
        print(row)
    print('=' * 76)


RAIN_CONDITIONS = {
    'Patchy rain possible',
    'Light rain',
    'Light rain shower',
    'Moderate rain at times',
    'Moderate rain',
    'Heavy rain at times',
    'Heavy rain',
    'Moderate or heavy rain shower',
    'Torrential rain shower',
    'Thundery outbreaks possible',
    'Patchy light rain',
    'Patchy light rain with thunder',
    'Patchy moderate rain',
    'Moderate or heavy rain with thunder',
}

# ============================================================
# Các tập condition theo loại
# ============================================================

RAIN_CONDITIONS = {
    'Patchy rain possible',
    'Light rain',
    'Light rain shower',
    'Moderate rain at times',
    'Moderate rain',
    'Heavy rain at times',
    'Heavy rain',
    'Moderate or heavy rain shower',
    'Torrential rain shower',
    'Thundery outbreaks possible',
    'Patchy light rain',
    'Patchy light rain with thunder',
    'Patchy moderate rain',
    'Moderate or heavy rain with thunder',
}

CLEAR_CONDITIONS  = {'Sunny', 'Clear'}
CLOUDY_CONDITIONS = {'Partly Cloudy', 'Partly cloudy', 'Cloudy', 'Overcast'}
MIST_CONDITIONS   = {'Mist', 'Fog', 'Freezing fog'}


def _is_daytime(hour: int) -> bool:
    return 6 <= hour < 18


def _cloud_to_condition(cloud: float, hour: int) -> str:
    """
    Chuyển cloud % → condition dựa trên WMO standard + giờ trong ngày.
    """
    if cloud < 25:
        return 'Sunny' if _is_daytime(hour) else 'Clear'
    elif cloud < 50:
        return 'Partly Cloudy'
    elif cloud < 75:
        return 'Cloudy'
    else:
        return 'Overcast'


def _precip_to_rain_condition(precip: float) -> str:
    """
    Chuyển precipitation mm → loại mưa phù hợp.
    """
    if precip < 0.1:
        return 'Patchy rain possible'
    elif precip < 0.5:
        return 'Light rain shower'
    elif precip < 1.0:
        return 'Moderate rain at times'
    elif precip < 2.0:
        return 'Heavy rain at times'
    else:
        return 'Heavy rain at times'


def fix_consistency(forecast: list) -> list:
    """
    Override condition khi có mâu thuẫn rõ ràng giữa
    rain_probability, cloud, precipitation, visibility và giờ trong ngày.

    Rules theo thứ tự ưu tiên:
      1. Mist/Fog  : visibility thấp + humidity cao → override tất cả
      2. rain_prob=0 + precip nhỏ : dùng cloud+giờ → Sunny/Cloudy
      3. rain_prob=100 + precip lớn: dùng precip   → loại mưa cụ thể
      4. rain_prob=45             : giữ nguyên classifier
    """
    fixed_count = 0

    for step in forecast:
        rp         = step.get('rain_probability', 0)
        cond       = step.get('condition', '')
        cloud      = step.get('cloud', 50)
        precip     = step.get('precipitation', 0)
        humidity   = step.get('humidity', 70)
        visibility = step.get('visibility', 10)
        hour       = pd.to_datetime(step['timestamp']).hour

        original = cond

        # --------------------------------------------------------
        # Rule 1: Mist / Fog
        # visibility < 2km VÀ humidity > 90% → Mist bất kể rain_prob
        # --------------------------------------------------------
        if visibility < 2.0 and humidity > 90:
            step['condition'] = 'Mist'

        # --------------------------------------------------------
        # Rule 2: Không mưa (rain_prob=0 VÀ precip nhỏ)
        # → 2 model đều nói không mưa → dùng cloud + giờ
        # --------------------------------------------------------
        elif rp == 0 and precip < 0.1:
            if cond in RAIN_CONDITIONS or cond in MIST_CONDITIONS:
                step['condition'] = _cloud_to_condition(cloud, hour)

            # Kiểm tra thêm: ban đêm không thể Sunny
            elif cond in CLEAR_CONDITIONS and not _is_daytime(hour):
                step['condition'] = 'Clear'

            # Kiểm tra: ban ngày cloud < 25% nhưng classifier cho Cloudy
            elif _is_daytime(hour) and cloud < 25 and cond not in CLEAR_CONDITIONS:
                if cloud < 25:
                    step['condition'] = 'Sunny'

        # --------------------------------------------------------
        # Rule 3: Chắc chắn mưa (rain_prob=100 VÀ precip đáng kể)
        # → condition phải là rain condition cụ thể
        # --------------------------------------------------------
        elif rp == 100 and precip >= 0.1:
            if cond not in RAIN_CONDITIONS:
                # Classifier sai hoàn toàn → dùng precip để chọn loại mưa
                step['condition'] = _precip_to_rain_condition(precip)
            else:
                # Classifier đúng loại mưa, nhưng kiểm tra độ nặng có hợp lý không
                current_is_heavy = 'Heavy' in cond or 'Torrential' in cond
                current_is_light = 'Light' in cond or 'Patchy' in cond

                if precip >= 1.5 and current_is_light:
                    # Precip lớn nhưng classifier nói light → nâng lên
                    step['condition'] = _precip_to_rain_condition(precip)
                elif precip < 0.1 and current_is_heavy:
                    # Precip nhỏ nhưng classifier nói heavy → hạ xuống
                    step['condition'] = _precip_to_rain_condition(precip)

        # --------------------------------------------------------
        # Rule 4: rain_prob=45 (uncertain)
        # Chỉ fix các trường hợp rõ ràng sai
        # --------------------------------------------------------
        elif rp == 45:
            # Ban đêm không thể Sunny
            if cond == 'Sunny' and not _is_daytime(hour):
                step['condition'] = 'Clear'
            # Cloud rất thấp mà classifier nói Cloudy/Overcast
            elif cloud < 20 and cond in ('Cloudy', 'Overcast'):
                step['condition'] = _cloud_to_condition(cloud, hour)

        # --------------------------------------------------------
        # Rule chung: Sunny không thể xảy ra ban đêm (mọi rain_prob)
        # --------------------------------------------------------
        if step['condition'] == 'Sunny' and not _is_daytime(hour):
            step['condition'] = 'Clear'

        if step['condition'] != original:
            fixed_count += 1

    if fixed_count:
        print(f"  [CONSISTENCY] Fixed {fixed_count} inconsistent steps")
    return forecast


def save_json(pred_tcn, cols_tcn, pred_hard, cols_hard,
              horizon, last_ts, output_path,
              tcn_path, tcn_hard_path,
              rain_prob_ckpt=None,
              classifier_ckpt=None):

    # Bước 1: build forecast steps từ 2 TCN
    forecast = []
    for hi in range(horizon):
        entry = {
            'timestamp':    (last_ts + timedelta(hours=hi + 1)).isoformat(),
            'horizon_step': hi + 1,
        }
        for ti, c in enumerate(cols_tcn):
            entry[c] = round(float(pred_tcn[hi, ti]), 4)
        for ti, c in enumerate(cols_hard):
            entry[c] = round(float(pred_hard[hi, ti]), 4)
        forecast.append(entry)

    # Bước 2: thêm rain_probability từ classifier
    if rain_prob_ckpt is not None:
        rain_probs = predict_rain_probabilities(rain_prob_ckpt, forecast)
        for hi, rp in enumerate(rain_probs):
            forecast[hi]['rain_probability'] = rp
        print(f"  [RAIN_PROB] Added rain_probability to {len(forecast)} steps")
    else:
        # Fallback: điền 0.0 nếu không có classifier
        for hi in range(len(forecast)):
            forecast[hi]['rain_probability'] = 0.0
        print("  [RAIN_PROB] No classifier — rain_probability defaulted to 0.0")

    # Bước 3: thêm condition (cần rain_probability đã có trong step)
    if classifier_ckpt is not None:
        conditions = predict_conditions(classifier_ckpt, forecast)
        for hi, cond in enumerate(conditions):
            forecast[hi]['condition'] = cond
        forecast = fix_consistency(forecast)

    # Bước 4: build result dict
    all_target_cols = (
        cols_tcn
        + cols_hard
        + ['rain_probability']
        + (['condition'] if classifier_ckpt else [])
    )

    result = {
        'generated_at':        datetime.now().isoformat(),
        'based_on_data_until': last_ts.isoformat(),
        'horizon_hours':       horizon,
        'model_used':          'TCN + TCN-Hard + RainProbClassifier',
        'models': {
            'tcn':                  tcn_path,
            'tcn_hard':             tcn_hard_path,
            'rain_prob_classifier': rain_prob_ckpt is not None,
            'condition_classifier': classifier_ckpt is not None,
        },
        'targets': {
            'tcn':      cols_tcn,
            'tcn_hard': cols_hard,
        },
        'target_cols': all_target_cols,
        'forecast':    forecast,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\n[INFO] Result saved: {output_path}')
    return result


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Dual TCN Inference: TCNForecaster + HardTargetForecaster'
    )
    parser.add_argument('--tcn',      type=str, required=True,
                        help='Path TCNForecaster checkpoint (.pth)')
    parser.add_argument('--tcn_hard', type=str, required=True,
                        help='Path HardTargetForecaster checkpoint (.pth)')
    parser.add_argument('--csv',      type=str, required=True,
                        help='CSV with at least lookback rows')
    parser.add_argument('--output',   type=str, default='forecast_result.json')
    parser.add_argument('--device',   type=str, default='cpu',
                        help='cpu or cuda')
    parser.add_argument('--classifier',          type=str, default=None,
                        help='Path to condition_classifier.pkl (optional)')
    parser.add_argument('--rain_prob_classifier', type=str, default=None,
                        help='Path to rain_prob_classifier.pkl (outputs 0/45/100%%)')
    args = parser.parse_args()

    print('=' * 76)
    print('  INFERENCE - Dual TCN Weather Forecast')
    print('=' * 76)

    # --- STEP 1: TCNForecaster ---
    print(f'\n[STEP 1] TCNForecaster  ->  {args.tcn}')
    pred_tcn, cols_tcn, h_tcn, ts_tcn = predict_tcn(args.tcn, args.csv, args.device)
    pred_tcn = postprocess(pred_tcn, cols_tcn)
    print(f'   targets : {cols_tcn}')
    print(f'   horizon : {h_tcn}h')

    # --- STEP 2: HardTargetForecaster ---
    print(f'\n[STEP 2] HardTargetForecaster  ->  {args.tcn_hard}')
    pred_hard, cols_hard, h_hard, ts_hard = predict_tcn_hard(
        args.tcn_hard, args.csv, args.device
    )
    pred_hard = postprocess(pred_hard, cols_hard)
    print(f'   targets : {cols_hard}')
    print(f'   horizon : {h_hard}h')

    # --- Align horizon ---
    horizon = min(h_tcn, h_hard)
    if h_tcn != h_hard:
        print(f'  [WARN] Horizon mismatch: TCN={h_tcn}h, Hard={h_hard}h -> using {horizon}h')
    pred_tcn  = pred_tcn[:horizon]
    pred_hard = pred_hard[:horizon]
    last_ts   = ts_tcn

    # --- STEP 3: Print tables ---
    print_table(pred_tcn,  cols_tcn,  horizon, last_ts,
                title='TCN - temperature / humidity / wind_speed / ...')
    print_table(pred_hard, cols_hard, horizon, last_ts,
                title='TCN-Hard - wind_direction / precipitation / cloud')

    # --- STEP 3b: Load rain probability classifier ---
    rain_prob_ckpt = None
    if args.rain_prob_classifier:
        print(f'\n[STEP 3b] Rain Prob Classifier  ->  {args.rain_prob_classifier}')
        rain_prob_ckpt = load_rain_prob_classifier(args.rain_prob_classifier)

    # --- STEP 3c: Load condition classifier ---
    classifier_ckpt = None
    if args.classifier:
        print(f'\n[STEP 3c] Condition Classifier  ->  {args.classifier}')
        classifier_ckpt = load_condition_classifier(args.classifier)

    # --- STEP 4: Save JSON ---
    result = save_json(
        pred_tcn, cols_tcn,
        pred_hard, cols_hard,
        horizon, last_ts,
        args.output,
        args.tcn, args.tcn_hard,
        rain_prob_ckpt=rain_prob_ckpt,
        classifier_ckpt=classifier_ckpt,
    )

    # --- Summary h+1 ---
    print('\n' + '=' * 76)
    print('  SUMMARY - NEXT 1 HOUR (h+1)')
    print('=' * 76)
    step1 = result['forecast'][0]
    for col in result['target_cols']:
        u     = UNITS.get(col, '')
        label = f'{col} ({u})' if u else col
        val   = step1.get(col, 'N/A')
        if isinstance(val, float):
            print(f'  {label:<30s}: {val:.3f}')
        else:
            print(f'  {label:<30s}: {val}')
    print('=' * 76)


if __name__ == '__main__':
    main()