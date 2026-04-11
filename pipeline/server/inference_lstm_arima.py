"""
=============================================================================
Inference Script — LSTM + ARIMA Ensemble
=============================================================================
Nhận vào file CSV chứa tối thiểu 48 hàng gần nhất,
xuất ra dự báo 6h tới.

Cách dùng:
    # Ensemble (khuyến nghị)
    python inference_lstm_arima.py \
        --lstm  lstm_model.pth \
        --arima arima_SGN.pkl \
        --csv   latest_48h.csv \
        --ensemble_json ensemble_lstm_SGN_metrics.json \
        --output forecast_result_lstm.json

    # Chỉ LSTM
    python inference_lstm_arima.py --lstm lstm_model.pth --csv latest_48h.csv

    # Chỉ ARIMA
    python inference_lstm_arima.py --arima arima_SGN.pkl --csv latest_48h.csv
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

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

if TORCH_OK:
    class LSTMForecaster(nn.Module):
        def __init__(self, num_features, num_targets, horizon,
                     hidden_size=128, num_layers=2, dropout=0.2):
            super().__init__()
            self.horizon     = horizon
            self.num_targets = num_targets
            self.hidden_size = hidden_size
            self.num_layers  = num_layers
            self.lstm = nn.LSTM(
                input_size  = num_features,
                hidden_size = hidden_size,
                num_layers  = num_layers,
                batch_first = True,
                dropout     = dropout if num_layers > 1 else 0.0,
            )
            # Layer normalization — khớp với train_lstm.py
            self.layer_norm = nn.LayerNorm(hidden_size)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, horizon * num_targets)
            )

        def forward(self, x):
            lstm_out, (h_n, c_n) = self.lstm(x)
            out = lstm_out[:, -1, :]   # timestep cuối
            out = self.layer_norm(out) # normalize — giống train_lstm.py
            out = self.fc(out)
            return out.view(-1, self.horizon, self.num_targets)


# ===========================================================================
# 1. Load input CSV
# ===========================================================================

def load_input_csv(csv_path, feature_cols_expected, lookback):
    df = pd.read_csv(csv_path)
    print(f"[INFO] Đọc {len(df)} hàng từ {csv_path}")

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    if len(df) < lookback:
        raise ValueError(
            f"[ERROR] File CSV chỉ có {len(df)} hàng, "
            f"cần ít nhất {lookback} hàng."
        )

    df = df.tail(lookback).reset_index(drop=True)
    print(f"[INFO] Dùng {lookback} hàng gần nhất để dự báo.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear').ffill().bfill()
    return df


def get_last_timestamp(df):
    if 'timestamp' in df.columns:
        return pd.to_datetime(df['timestamp'].iloc[-1])
    return datetime.now()


# ===========================================================================
# 2. LSTM inference
# ===========================================================================

def predict_lstm(lstm_path, csv_path, device_str='cpu'):
    if not TORCH_OK:
        raise ImportError("pip install torch")

    device = torch.device(device_str)
    ckpt   = torch.load(lstm_path, map_location=device, weights_only=False)
    cfg    = ckpt['model_config']

    model = LSTMForecaster(
        num_features = cfg['num_features'],
        num_targets  = cfg['num_targets'],
        horizon      = cfg['horizon'],
        hidden_size  = cfg.get('hidden_size', 128),
        num_layers   = cfg.get('num_layers', 2),
        dropout      = cfg.get('dropout', 0.2),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

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

    input_df    = load_input_csv(csv_path, feature_cols, lookback)
    features    = input_df[feature_cols].values.astype(np.float32)
    features_sc = feat_scaler.transform(features)

    x_tensor = torch.FloatTensor(features_sc).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy()   # (1, horizon, num_targets)

    pred_orig = tgt_scaler.inverse_transform(
        pred_scaled[0].reshape(-1, len(target_cols))
    ).reshape(horizon, len(target_cols))

    last_ts = get_last_timestamp(input_df)
    return pred_orig, target_cols, horizon, last_ts, lookback


# ===========================================================================
# 3. ARIMA inference
# ===========================================================================

def predict_arima(arima_path, csv_path):
    try:
        from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
    except ImportError:
        raise ImportError("pip install statsmodels")

    with open(arima_path, 'rb') as f:
        ckpt = pickle.load(f)

    cfg         = ckpt['model_config']
    target_cols = ckpt['target_cols']
    lookback    = cfg['lookback']
    horizon     = cfg['horizon']
    orders      = cfg['orders']

    tgt_scaler         = StandardScaler()
    tgt_scaler.mean_   = np.array(ckpt['target_scaler_mean'])
    tgt_scaler.scale_  = np.array(ckpt['target_scaler_scale'])
    tgt_scaler.var_    = tgt_scaler.scale_ ** 2
    tgt_scaler.n_features_in_  = len(tgt_scaler.mean_)

    input_df     = load_input_csv(csv_path, target_cols, lookback)
    targets      = input_df[target_cols].values.astype(np.float32)
    targets_sc   = tgt_scaler.transform(targets)
    pred_scaled  = np.zeros((horizon, len(target_cols)), dtype=np.float32)

    for ti, col in enumerate(target_cols):
        series  = targets_sc[:, ti].astype(np.float64)
        p, d, q = tuple(orders.get(col, [2, 1, 2]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = SM_ARIMA(series, order=(p, d, q)).fit(
                    method_kwargs={"warn_convergence": False})
                fc  = res.forecast(steps=horizon)
            except Exception as e:
                print(f"  [WARN] ARIMA '{col}': {e} — dùng last value")
                fc  = np.full(horizon, series[-1])
        pred_scaled[:, ti] = fc[:horizon]

    pred_orig = tgt_scaler.inverse_transform(pred_scaled)
    last_ts   = get_last_timestamp(input_df)
    return pred_orig, target_cols, horizon, last_ts, lookback


# ===========================================================================
# 4. Load weights từ ensemble JSON
# ===========================================================================

def load_weights_from_json(json_path, target_cols):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    weights_lstm  = data.get('weights', {}).get('lstm',  {})
    weights_arima = data.get('weights', {}).get('arima', {})

    missing = [c for c in target_cols
               if c not in weights_lstm or c not in weights_arima]
    if missing:
        print(f"  [WARN] Cột thiếu trong JSON: {missing} → dùng 0.5/0.5")
        for c in missing:
            weights_lstm[c]  = 0.5
            weights_arima[c] = 0.5

    strategy = data.get('config', {}).get('strategy', 'unknown')
    print(f"  [INFO] Đọc weights từ: {json_path} (strategy={strategy})")
    print(f"\n  {'Column':<25s} {'w_LSTM':>8s} {'w_ARIMA':>8s}")
    print("  " + "-" * 42)
    for col in target_cols:
        print(f"  {col:<25s} {weights_lstm.get(col,0.5):>8.4f} "
              f"{weights_arima.get(col,0.5):>8.4f}")
    return weights_lstm, weights_arima


# ===========================================================================
# 5. Ensemble
# ===========================================================================

def ensemble_predictions(pred_lstm, pred_arima, target_cols,
                         weights_lstm=None, weights_arima=None,
                         w_lstm_fixed=0.6, w_arima_fixed=0.4):
    if pred_lstm is None:
        return pred_arima
    if pred_arima is None:
        return pred_lstm

    horizon, num_targets = pred_lstm.shape
    preds = np.zeros_like(pred_lstm)

    for ti, col in enumerate(target_cols):
        if weights_lstm is not None and weights_arima is not None:
            wl = weights_lstm.get(col, 0.5)
            wa = weights_arima.get(col, 0.5)
        else:
            total = w_lstm_fixed + w_arima_fixed
            wl    = w_lstm_fixed  / total
            wa    = w_arima_fixed / total
        preds[:, ti] = pred_lstm[:, ti] * wl + pred_arima[:, ti] * wa

    if weights_lstm is not None:
        print(f"[INFO] Ensemble: per-column weights từ JSON")
    else:
        print(f"[INFO] Ensemble: LSTM×{w_lstm_fixed:.2f} + ARIMA×{w_arima_fixed:.2f} (fixed)")
    return preds


# ===========================================================================
# 6. Output
# ===========================================================================

UNITS = {
    'temperature': '°C', 'feels_like': '°C', 'dewpoint': '°C',
    'humidity': '%', 'wind_speed': 'km/h', 'pressure': 'hPa',
    'precipitation': 'mm', 'uv_index': '', 'visibility': 'km',
    'rain_probability': '%', 'cloud': '%',
}


def print_forecast_table(preds, target_cols, horizon, last_ts):
    print("\n" + "=" * 70)
    print(f"  DỰ BÁO THỜI TIẾT — {horizon}h tới  (LSTM + ARIMA)")
    print(f"  Dựa trên dữ liệu đến: {last_ts.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    header = f"  {'Thời điểm':<20s}"
    for col in target_cols:
        unit  = UNITS.get(col, '')
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
              model_used, w_lstm, w_arima):
    forecast_list = []
    for hi in range(horizon):
        ts    = last_ts + timedelta(hours=hi + 1)
        entry = {'timestamp': ts.isoformat(), 'horizon_step': hi + 1}
        for ti, col in enumerate(target_cols):
            entry[col] = round(float(preds[hi, ti]), 4)
        forecast_list.append(entry)

    result = {
        'generated_at':          datetime.now().isoformat(),
        'based_on_data_until':   last_ts.isoformat(),
        'horizon_hours':         horizon,
        'model_used':            model_used,
        'weights':               {'lstm': w_lstm, 'arima': w_arima},
        'target_cols':           target_cols,
        'forecast':              forecast_list,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Kết quả lưu tại: {output_path}")
    return result


# ===========================================================================
# 7. Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inference: LSTM + ARIMA Ensemble"
    )
    parser.add_argument('--lstm',  type=str, default=None,
                        help='Path to LSTM checkpoint (.pth)')
    parser.add_argument('--arima', type=str, default=None,
                        help='Path to ARIMA checkpoint (.pkl)')
    parser.add_argument('--csv',   type=str, required=True,
                        help='CSV chứa ít nhất lookback hàng gần nhất')

    parser.add_argument('--ensemble_json', type=str, default=None,
                        help='File JSON từ ensemble_lstm_arima.py '
                             '(vd: ensemble_lstm_SGN_metrics.json)')
    parser.add_argument('--w_lstm',  type=float, default=0.6,
                        help='Trọng số LSTM cố định (dùng khi không có --ensemble_json)')
    parser.add_argument('--w_arima', type=float, default=0.4,
                        help='Trọng số ARIMA cố định (dùng khi không có --ensemble_json)')

    parser.add_argument('--output', type=str, default='forecast_result_lstm.json')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    if args.lstm is None and args.arima is None:
        parser.error("Cần ít nhất --lstm hoặc --arima")

    print("=" * 70)
    print("  INFERENCE — LSTM + ARIMA Ensemble")
    print("=" * 70)

    # LSTM inference
    pred_lstm    = None
    target_cols  = None
    horizon      = None
    last_ts      = None

    if args.lstm:
        print(f"\n[STEP 1] LSTM inference...")
        pred_lstm, target_cols, horizon, last_ts, _ = predict_lstm(
            args.lstm, args.csv, args.device)
        print(f"  ✓ LSTM done. Shape: {pred_lstm.shape}")

    # ARIMA inference
    pred_arima = None

    if args.arima:
        print(f"\n[STEP 2] ARIMA inference...")
        pred_arima, arima_targets, arima_horizon, arima_last_ts, _ = predict_arima(
            args.arima, args.csv)
        print(f"  ✓ ARIMA done. Shape: {pred_arima.shape}")

        if target_cols is None:
            target_cols = arima_targets
            horizon     = arima_horizon
            last_ts     = arima_last_ts
        elif target_cols != arima_targets:
            common      = [c for c in target_cols if c in arima_targets]
            print(f"  [WARN] Target cols lệch — dùng intersection: {common}")
            lstm_idx    = [target_cols.index(c)   for c in common]
            arima_idx   = [arima_targets.index(c) for c in common]
            pred_lstm   = pred_lstm[:, lstm_idx]  if pred_lstm  is not None else None
            pred_arima  = pred_arima[:, arima_idx]
            target_cols = common

    # Đọc weights từ JSON (nếu có)
    weights_lstm  = None
    weights_arima = None

    if args.ensemble_json and args.lstm and args.arima:
        if not os.path.exists(args.ensemble_json):
            print(f"[WARN] Không tìm thấy {args.ensemble_json} — dùng fixed weights.")
        else:
            print(f"\n[STEP 3] Đọc trọng số từ {args.ensemble_json}...")
            weights_lstm, weights_arima = load_weights_from_json(
                args.ensemble_json, target_cols)

    # Ensemble
    step = 4 if weights_lstm else 3
    print(f"\n[STEP {step}] Ensemble...")
    pred_final = ensemble_predictions(
        pred_lstm     = pred_lstm,
        pred_arima    = pred_arima,
        target_cols   = target_cols,
        weights_lstm  = weights_lstm,
        weights_arima = weights_arima,
        w_lstm_fixed  = args.w_lstm,
        w_arima_fixed = args.w_arima,
    )

    model_used = (
        'LSTM+ARIMA ensemble' if (args.lstm and args.arima)
        else ('LSTM' if args.lstm else 'ARIMA')
    )

    print_forecast_table(pred_final, target_cols, horizon, last_ts)

    if weights_lstm:
        final_w_lstm  = {c: weights_lstm.get(c, 0.5)  for c in target_cols}
        final_w_arima = {c: weights_arima.get(c, 0.5) for c in target_cols}
    else:
        total = args.w_lstm + args.w_arima
        final_w_lstm  = {c: round(args.w_lstm  / total, 4) for c in (target_cols or [])}
        final_w_arima = {c: round(args.w_arima / total, 4) for c in (target_cols or [])}

    save_json(
        preds       = pred_final,
        target_cols = target_cols,
        horizon     = horizon,
        last_ts     = last_ts,
        output_path = args.output,
        model_used  = model_used,
        w_lstm      = final_w_lstm  if args.lstm  else 0,
        w_arima     = final_w_arima if args.arima else 0,
    )


if __name__ == '__main__':
    main()








  




 














