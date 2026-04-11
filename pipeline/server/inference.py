"""
=============================================================================
Inference Script — TCN + ARIMA Ensemble
=============================================================================
Nhận vào file CSV chứa tối thiểu 48 hàng gần nhất (48h thu thập),
xuất ra dự báo 6h tới cho tất cả target columns.

Cách dùng:
    # Dùng TCN đơn
    python inference.py --tcn my_model.pth --csv latest_48h.csv

    # Dùng ARIMA đơn
    python inference.py --arima arima_SGN.pkl --csv latest_48h.csv

    # Dùng cả hai (ensemble)
    python inference.py --tcn my_model.pth --arima arima_SGN.pkl --csv latest_48h.csv

    # Chỉ định trọng số
    python inference.py --tcn my_model.pth --arima arima_SGN.pkl \
                        --csv latest_48h.csv --w_tcn 0.7 --w_arima 0.3

Output:
    In ra terminal bảng dự báo 6h tới
    Lưu file forecast_result.json (dùng cho Digital Twin / API)
=============================================================================
"""

import argparse
import json
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. TCN definition — giống train_new.py để load được .pth
# ============================================================
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

if TORCH_OK:
    class CausalConv1d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                  padding=self.padding, dilation=dilation)
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
            self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                               if in_channels != out_channels else None)
        def forward(self, x):
            residual = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.dropout(out)
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.dropout(out)
            if self.downsample is not None:
                residual = self.downsample(residual)
            return self.relu(out + residual)

    class TCNForecaster(nn.Module):
        def __init__(self, num_features, num_targets, horizon,
                     num_channels, kernel_size, dropout):
            super().__init__()
            self.horizon     = horizon
            self.num_targets = num_targets
            layers = []
            for i in range(len(num_channels)):
                in_ch  = num_features if i == 0 else num_channels[i - 1]
                layers.append(TemporalBlock(in_ch, num_channels[i],
                                            kernel_size, 2 ** i, dropout))
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


# ============================================================
# 2. Load data — nhận file CSV 48h gần nhất
# ============================================================

def load_input_csv(csv_path, feature_cols_expected, lookback):
    """
    Đọc CSV đầu vào, kiểm tra đủ lookback hàng, trả về DataFrame.
    """
    df = pd.read_csv(csv_path)
    print(f"[INFO] Đọc {len(df)} hàng từ {csv_path}")

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    if len(df) < lookback:
        raise ValueError(
            f"[ERROR] File CSV chỉ có {len(df)} hàng, "
            f"cần ít nhất {lookback} hàng (= {lookback}h dữ liệu).\n"
            f"Hãy thu thập thêm hoặc giảm --lookback."
        )

    # Lấy lookback hàng cuối cùng
    df = df.tail(lookback).reset_index(drop=True)
    print(f"[INFO] Dùng {lookback} hàng gần nhất để dự báo.")

    # Xử lý thiếu
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear').ffill().bfill()

    return df


def build_feature_array(df, feature_cols):
    """Trích feature array từ DataFrame."""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[ERROR] Các cột feature thiếu trong CSV: {missing}\n"
            f"Các cột hiện có: {list(df.columns)}"
        )
    return df[feature_cols].values.astype(np.float32)


def get_last_timestamp(df):
    if 'timestamp' in df.columns:
        return pd.to_datetime(df['timestamp'].iloc[-1])
    return datetime.now()


# ============================================================
# 3. TCN inference
# ============================================================

def predict_tcn(tcn_path, df, device_str='cpu'):
    if not TORCH_OK:
        raise ImportError("PyTorch chưa được cài. pip install torch")

    device = torch.device(device_str)
    ckpt   = torch.load(tcn_path, map_location=device, weights_only=False)
    cfg    = ckpt['model_config']

    # Rebuild model
    model = TCNForecaster(
        num_features = cfg['num_features'],
        num_targets  = cfg['num_targets'],
        horizon      = cfg['horizon'],
        num_channels = cfg['num_channels'],
        kernel_size  = cfg['kernel_size'],
        dropout      = cfg['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Rebuild scalers
    feat_scaler        = StandardScaler()
    feat_scaler.mean_  = np.array(ckpt['feature_scaler_mean'])
    feat_scaler.scale_ = np.array(ckpt['feature_scaler_scale'])
    feat_scaler.var_   = feat_scaler.scale_ ** 2
    feat_scaler.n_features_in_ = len(feat_scaler.mean_)

    tgt_scaler         = StandardScaler()
    tgt_scaler.mean_   = np.array(ckpt['target_scaler_mean'])
    tgt_scaler.scale_  = np.array(ckpt['target_scaler_scale'])
    tgt_scaler.var_    = tgt_scaler.scale_ ** 2
    tgt_scaler.n_features_in_  = len(tgt_scaler.mean_)

    feature_cols = ckpt['feature_cols']
    target_cols  = ckpt['target_cols']
    lookback     = cfg['lookback']
    horizon      = cfg['horizon']

    # Load & prep input
    input_df      = load_input_csv(df if isinstance(df, str) else df,
                                   feature_cols, lookback)
    features      = build_feature_array(input_df, feature_cols)
    features_sc   = feat_scaler.transform(features)   # (lookback, num_features)

    x_tensor = torch.FloatTensor(features_sc).unsqueeze(0).to(device)  # (1, lookback, F)

    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy()   # (1, horizon, num_targets)

    # Inverse transform
    pred_orig = tgt_scaler.inverse_transform(
        pred_scaled[0].reshape(-1, len(target_cols))
    ).reshape(horizon, len(target_cols))              # (horizon, num_targets)

    last_ts = get_last_timestamp(input_df)
    return pred_orig, target_cols, horizon, last_ts, lookback


# ============================================================
# 4. ARIMA inference
# ============================================================

def predict_arima(arima_path, df):
    try:
        from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
    except ImportError:
        raise ImportError("statsmodels chưa được cài. pip install statsmodels")

    with open(arima_path, 'rb') as f:
        ckpt = pickle.load(f)

    cfg         = ckpt['model_config']
    target_cols = ckpt['target_cols']
    lookback    = cfg['lookback']
    horizon     = cfg['horizon']
    orders      = cfg['orders']

    # Rebuild scaler
    tgt_scaler         = StandardScaler()
    tgt_scaler.mean_   = np.array(ckpt['target_scaler_mean'])
    tgt_scaler.scale_  = np.array(ckpt['target_scaler_scale'])
    tgt_scaler.var_    = tgt_scaler.scale_ ** 2
    tgt_scaler.n_features_in_  = len(tgt_scaler.mean_)

    input_df = load_input_csv(df if isinstance(df, str) else df,
                              target_cols, lookback)
    targets  = input_df[target_cols].values.astype(np.float32)

    # Scale
    targets_sc   = tgt_scaler.transform(targets)      # (lookback, num_targets)
    pred_scaled  = np.zeros((horizon, len(target_cols)), dtype=np.float32)

    for ti, col in enumerate(target_cols):
        series = targets_sc[:, ti].astype(np.float64)
        p, d, q = tuple(orders.get(col, [2, 1, 2]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = SM_ARIMA(series, order=(p, d, q)).fit(
                    method_kwargs={"warn_convergence": False})
                fc  = res.forecast(steps=horizon)
            except Exception as e:
                print(f"  [WARN] ARIMA fit lỗi cho '{col}': {e} — dùng last value")
                fc  = np.full(horizon, series[-1])
        pred_scaled[:, ti] = fc[:horizon]

    # Inverse transform
    pred_orig = tgt_scaler.inverse_transform(pred_scaled)  # (horizon, num_targets)

    last_ts = get_last_timestamp(input_df)
    return pred_orig, target_cols, horizon, last_ts, lookback


# ============================================================
# 5. Ensemble 2 predictions
# ============================================================

def load_weights_from_json(json_path, target_cols):
    """
    Đọc trọng số per-column từ file ensemble_SGN_metrics.json.
    Trả về 2 dict: weights_tcn[col], weights_arima[col]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    weights_tcn   = data.get('weights', {}).get('tcn',   {})
    weights_arima = data.get('weights', {}).get('arima', {})

    # Kiểm tra có đủ cột không
    missing = [c for c in target_cols
               if c not in weights_tcn or c not in weights_arima]
    if missing:
        print(f"  [WARN] Các cột sau không có trong JSON weights: {missing}")
        print(f"  [WARN] Dùng 0.5/0.5 cho các cột thiếu.")
        for c in missing:
            weights_tcn[c]   = 0.5
            weights_arima[c] = 0.5

    strategy = data.get('config', {}).get('strategy', 'unknown')
    print(f"  [INFO] Đọc weights từ: {json_path}")
    print(f"  [INFO] Strategy gốc  : {strategy}")
    print(f"\n  {'Column':<25s} {'w_TCN':>8s} {'w_ARIMA':>8s}")
    print("  " + "-" * 42)
    for col in target_cols:
        wt = weights_tcn.get(col, 0.5)
        wa = weights_arima.get(col, 0.5)
        print(f"  {col:<25s} {wt:>8.4f} {wa:>8.4f}")

    return weights_tcn, weights_arima


def ensemble_predictions(pred_tcn, pred_arima, target_cols,
                         weights_tcn=None, weights_arima=None,
                         w_tcn_fixed=0.6, w_arima_fixed=0.4):
    """
    Kết hợp dự báo TCN + ARIMA.

    Nếu có weights_tcn/weights_arima (đọc từ JSON) → dùng per-column weights.
    Nếu không → dùng w_tcn_fixed / w_arima_fixed cho tất cả cột.

    pred_tcn / pred_arima: (horizon, num_targets)
    """
    if pred_tcn is None:
        return pred_arima
    if pred_arima is None:
        return pred_tcn

    horizon, num_targets = pred_tcn.shape
    preds = np.zeros_like(pred_tcn)

    for ti, col in enumerate(target_cols):
        if weights_tcn is not None and weights_arima is not None:
            # Per-column weights từ ensemble JSON
            wt = weights_tcn.get(col, 0.5)
            wa = weights_arima.get(col, 0.5)
        else:
            # Fixed weights từ argument
            total = w_tcn_fixed + w_arima_fixed
            wt    = w_tcn_fixed  / total
            wa    = w_arima_fixed / total

        preds[:, ti] = pred_tcn[:, ti] * wt + pred_arima[:, ti] * wa

    if weights_tcn is not None:
        print(f"[INFO] Ensemble: per-column weights từ JSON")
    else:
        print(f"[INFO] Ensemble: TCN×{w_tcn_fixed:.2f} + ARIMA×{w_arima_fixed:.2f} (fixed)")

    return preds


# ============================================================
# 6. Output — in bảng + lưu JSON
# ============================================================

UNITS = {
    'temperature': '°C', 'feels_like': '°C', 'dewpoint': '°C',
    'humidity': '%',
    'wind_speed': 'km/h', 'gust_speed': 'km/h',
    'wind_direction': '°',
    'pressure': 'hPa',
    'precipitation': 'mm',
    'rain_probability': '%',
    'uv_index': '',
    'visibility': 'km',
    'cloud': '%',
}


def print_forecast_table(preds, target_cols, horizon, last_ts):
    """In bảng dự báo ra terminal."""
    print("\n" + "=" * 70)
    print(f"  DỰ BÁO THỜI TIẾT — {horizon}h tới")
    print(f"  Dựa trên dữ liệu đến: {last_ts.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Header
    header = f"  {'Thời điểm':<20s}"
    for col in target_cols:
        unit = UNITS.get(col, '')
        label = f"{col}({unit})" if unit else col
        header += f"  {label:>15s}"
    print(header)
    print("  " + "-" * (18 + 17 * len(target_cols)))

    for hi in range(horizon):
        ts_str = (last_ts + timedelta(hours=hi + 1)).strftime('%Y-%m-%d %H:%M')
        row    = f"  {ts_str:<20s}"
        for ti in range(len(target_cols)):
            row += f"  {preds[hi, ti]:>15.2f}"
        print(row)

    print("=" * 70)


def save_json(preds, target_cols, horizon, last_ts, output_path,
              model_used, w_tcn, w_arima):
    """Lưu kết quả dự báo ra JSON cho Digital Twin / API đọc."""
    forecast_list = []
    for hi in range(horizon):
        ts = last_ts + timedelta(hours=hi + 1)
        entry = {
            'timestamp':    ts.isoformat(),
            'horizon_step': hi + 1,
        }
        for ti, col in enumerate(target_cols):
            entry[col] = round(float(preds[hi, ti]), 4)
        forecast_list.append(entry)

    result = {
        'generated_at':  datetime.now().isoformat(),
        'based_on_data_until': last_ts.isoformat(),
        'horizon_hours': horizon,
        'model_used':    model_used,
        'weights':       {'tcn': w_tcn, 'arima': w_arima},
        'target_cols':   target_cols,
        'forecast':      forecast_list,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Kết quả lưu tại: {output_path}")
    return result


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inference: TCN + ARIMA Ensemble — dự báo thời tiết 6h tới"
    )
    # Model paths
    parser.add_argument('--tcn',   type=str, default=None,
                        help='Path to TCN checkpoint (.pth)')
    parser.add_argument('--arima', type=str, default=None,
                        help='Path to ARIMA checkpoint (.pkl)')

    # Input data
    parser.add_argument('--csv', type=str, required=True,
                        help='CSV chứa ít nhất lookback hàng gần nhất (vd: latest_48h.csv)')

    # Ensemble weights — ưu tiên ensemble_json > w_tcn/w_arima thủ công
    parser.add_argument('--ensemble_json', type=str, default=None,
                        help='File JSON từ ensemble.py (vd: ensemble_SGN_metrics.json). '
                             'Tự đọc trọng số per-column, bỏ qua --w_tcn/--w_arima.')
    parser.add_argument('--w_tcn',   type=float, default=0.6,
                        help='Trọng số TCN cố định — chỉ dùng khi không có --ensemble_json')
    parser.add_argument('--w_arima', type=float, default=0.4,
                        help='Trọng số ARIMA cố định — chỉ dùng khi không có --ensemble_json')

    # Output
    parser.add_argument('--output', type=str, default='forecast_result.json',
                        help='File JSON lưu kết quả (mặc định: forecast_result.json)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device cho TCN: cpu hoặc cuda (mặc định: cpu)')

    args = parser.parse_args()

    if args.tcn is None and args.arima is None:
        parser.error("Cần ít nhất một trong --tcn hoặc --arima")

    print("=" * 70)
    print("  INFERENCE — TCN + ARIMA Ensemble")
    print("=" * 70)

    # --------------------------------------------------------
    # Chạy TCN
    # --------------------------------------------------------
    pred_tcn      = None
    target_cols   = None
    horizon       = None
    last_ts       = None

    if args.tcn:
        print(f"\n[STEP 1] TCN inference...")
        pred_tcn, target_cols, horizon, last_ts, lookback = predict_tcn(
            args.tcn, args.csv, args.device
        )
        print(f"  ✓ TCN done. Shape: {pred_tcn.shape}")

    # --------------------------------------------------------
    # Chạy ARIMA
    # --------------------------------------------------------
    pred_arima = None

    if args.arima:
        print(f"\n[STEP 2] ARIMA inference...")
        pred_arima, arima_targets, arima_horizon, arima_last_ts, _ = predict_arima(
            args.arima, args.csv
        )
        print(f"  ✓ ARIMA done. Shape: {pred_arima.shape}")

        # Nếu chỉ dùng ARIMA (không có TCN)
        if target_cols is None:
            target_cols = arima_targets
            horizon     = arima_horizon
            last_ts     = arima_last_ts

        # Align target_cols nếu 2 model dùng cột khác nhau
        elif target_cols != arima_targets:
            common = [c for c in target_cols if c in arima_targets]
            print(f"  [WARN] Target cols lệch — dùng intersection: {common}")
            tcn_idx   = [target_cols.index(c)   for c in common]
            arima_idx = [arima_targets.index(c) for c in common]
            pred_tcn   = pred_tcn[:, tcn_idx]   if pred_tcn   is not None else None
            pred_arima = pred_arima[:, arima_idx]
            target_cols = common

    # --------------------------------------------------------
    # Đọc weights từ ensemble JSON (nếu có)
    # --------------------------------------------------------
    weights_tcn   = None
    weights_arima = None

    if args.ensemble_json and args.tcn and args.arima:
        if not os.path.exists(args.ensemble_json):
            print(f"[WARN] Không tìm thấy {args.ensemble_json} — dùng fixed weights.")
        else:
            print(f"\n[STEP 3] Đọc trọng số từ {args.ensemble_json}...")
            weights_tcn, weights_arima = load_weights_from_json(
                args.ensemble_json, target_cols
            )
    elif args.ensemble_json and not (args.tcn and args.arima):
        print(f"[WARN] --ensemble_json chỉ có tác dụng khi dùng cả --tcn và --arima.")

    # --------------------------------------------------------
    # Ensemble
    # --------------------------------------------------------
    print(f"\n[STEP {'4' if weights_tcn else '3'}] Ensemble...")
    pred_final = ensemble_predictions(
        pred_tcn      = pred_tcn,
        pred_arima    = pred_arima,
        target_cols   = target_cols,
        weights_tcn   = weights_tcn,
        weights_arima = weights_arima,
        w_tcn_fixed   = args.w_tcn,
        w_arima_fixed = args.w_arima,
    )

    model_used = (
        'TCN+ARIMA ensemble' if (args.tcn and args.arima)
        else ('TCN' if args.tcn else 'ARIMA')
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    print_forecast_table(pred_final, target_cols, horizon, last_ts)

    # Lưu trọng số thực sự đã dùng vào JSON
    if weights_tcn:
        final_w_tcn   = {c: weights_tcn.get(c, 0.5)   for c in target_cols}
        final_w_arima = {c: weights_arima.get(c, 0.5) for c in target_cols}
    else:
        total = args.w_tcn + args.w_arima
        final_w_tcn   = {c: round(args.w_tcn   / total, 4) for c in (target_cols or [])}
        final_w_arima = {c: round(args.w_arima / total, 4) for c in (target_cols or [])}

    save_json(
        preds       = pred_final,
        target_cols = target_cols,
        horizon     = horizon,
        last_ts     = last_ts,
        output_path = args.output,
        model_used  = model_used,
        w_tcn       = final_w_tcn   if args.tcn   else 0,
        w_arima     = final_w_arima if args.arima else 0,
    )


if __name__ == '__main__':
    main()