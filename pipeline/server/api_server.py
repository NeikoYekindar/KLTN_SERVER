"""
=============================================================================
Forecast API Server  —  Flask + Local Mount (AWS EFS)
=============================================================================
Component 2: serve Flask API, đọc trực tiếp từ file trên volume mount.
Không cache, không interval — mỗi request đọc file tại chỗ.

Usage:
    python server/api_server.py

Env vars (optional):
    EFS_BASE   — EFS mount point (default /mnt/efs/fs1)
    API_HOST   — Flask host (default 0.0.0.0)
    API_PORT   — Flask port (default 5000)

Forecast step fields:
    timestamp, horizon_step,
    temperature, humidity, wind_speed, pressure, visibility,  ← TCN Model 1
    wind_direction, precipitation, cloud,                     ← TCN Hard Model
    rain_probability,                                         ← Rain Prob Classifier (0/45/100)
    condition                                                  ← Condition Classifier

Endpoints:
    GET  /api/forecast
    GET  /api/forecast/latest
    GET  /api/forecast/step/<n>
    GET  /api/forecast/history
    GET  /api/forecast/history/<run_id>
    GET  /api/current
    GET  /api/status
    POST /api/upload_csv
    GET  /api/data/rows           ← trả về toàn bộ last_48h.csv dạng JSON
    POST /api/forecast/custom     ← nhận rows đã sửa, lưu custom CSV, chạy lại inference
=============================================================================
"""

import csv
import glob
import io
import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

# ============================================================
# Config
# ============================================================

EFS_BASE      = os.environ.get('EFS_BASE', '/mnt/efs/fs1')
CSV_PATH      = f"{EFS_BASE}/data/last_48h.csv"
FORECAST_PATH = f"{EFS_BASE}/data/forecast_result.json"
HISTORY_DIR   = f"{EFS_BASE}/forecasts/history"
CUSTOM_DIR    = f"{EFS_BASE}/data/custom"

# Model paths — must match those passed to worker.py
TCN_MODEL          = os.environ.get('TCN_MODEL',            'models/tcn_model.pth')
TCN_HARD_MODEL     = os.environ.get('TCN_HARD_MODEL',       'models/tcn_hard_model.pth')
CLASSIFIER         = os.environ.get('CLASSIFIER',           'models/condition_classifier.pkl')
RAIN_PROB_CLS      = os.environ.get('RAIN_PROB_CLASSIFIER', 'models/rain_prob_classifier.pkl')
INFERENCE_SCRIPT   = os.environ.get('INFERENCE_SCRIPT',     'server/inference_dual_tcn.py')
INFERENCE_DEVICE   = os.environ.get('DEVICE',               'cpu')

_started_at     = datetime.now().isoformat()
_inference_lock = threading.Lock()

# ============================================================
# Flask App
# ============================================================

app = Flask(__name__)
CORS(app)


# ----------------------------------------------------------
# GET /api/forecast
# ----------------------------------------------------------
@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    if not os.path.exists(FORECAST_PATH):
        return jsonify({'error': 'No forecast available yet'}), 404
    try:
        with open(FORECAST_PATH, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------
# GET /api/forecast/latest
# ----------------------------------------------------------
@app.route('/api/forecast/latest', methods=['GET'])
def get_latest_step():
    if not os.path.exists(FORECAST_PATH):
        return jsonify({'error': 'No forecast available'}), 404
    try:
        with open(FORECAST_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        steps = data.get('forecast', [])
        if not steps:
            return jsonify({'error': 'No forecast steps'}), 404
        return jsonify(steps[0])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------
# GET /api/forecast/step/<n>
# ----------------------------------------------------------
@app.route('/api/forecast/step/<int:step>', methods=['GET'])
def get_forecast_step(step):
    if not os.path.exists(FORECAST_PATH):
        return jsonify({'error': 'No forecast available'}), 404
    try:
        with open(FORECAST_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for entry in data.get('forecast', []):
            if entry.get('horizon_step') == step:
                # Đảm bảo rain_probability luôn là một trong 3 giá trị hợp lệ
                rp = entry.get('rain_probability')
                if rp is not None and rp not in (0, 45, 100, 0.0, 45.0, 100.0):
                    entry['rain_probability_raw'] = rp
                    entry['rain_probability'] = min([0, 45, 100], key=lambda v: abs(v - rp))
                return jsonify(entry)
        return jsonify({'error': f'Step {step} not found (1–24)'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------
# GET /api/forecast/history
# ----------------------------------------------------------
@app.route('/api/forecast/history', methods=['GET'])
def get_forecast_history():
    limit = request.args.get('limit', 20, type=int)
    files = sorted(glob.glob(f"{HISTORY_DIR}/forecast_*.json"), reverse=True)[:limit]
    history = []
    for path in files:
        filename = os.path.basename(path)
        run_id   = filename.replace('.json', '').replace('forecast_', '', 1)
        history.append({'run_id': run_id, 'filename': filename})
    return jsonify({'count': len(history), 'history': history})


# ----------------------------------------------------------
# GET /api/forecast/history/<run_id>
# ----------------------------------------------------------
@app.route('/api/forecast/history/<run_id>', methods=['GET'])
def get_forecast_history_detail(run_id):
    path = f"{HISTORY_DIR}/forecast_{run_id}.json"
    if not os.path.exists(path):
        return jsonify({'error': f'run_id "{run_id}" not found'}), 404
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------
# GET /api/current
# ----------------------------------------------------------
@app.route('/api/current', methods=['GET'])
def get_current_data():
    if not os.path.exists(CSV_PATH):
        return jsonify({'error': 'No sensor data yet'}), 404
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return jsonify({'error': 'CSV is empty'}), 404
        return jsonify({
            'received_at': datetime.now().isoformat(),
            'num_rows':    len(rows),
            'last_row':    rows[-1],
            'columns':     list(rows[0].keys()),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------
# GET /api/status
# ----------------------------------------------------------
@app.route('/api/status', methods=['GET'])
def get_status():
    forecast_info = {}
    if os.path.exists(FORECAST_PATH):
        try:
            with open(FORECAST_PATH, 'r', encoding='utf-8') as f:
                fd = json.load(f)
            steps = fd.get('forecast', [])
            # Kiểm tra rain_probability có trong forecast không
            has_rain_prob = any('rain_probability' in s for s in steps)
            forecast_info = {
                'steps':        len(steps),
                'has_rain_probability': has_rain_prob,
                'generated_at': fd.get('generated_at'),
            }
        except Exception:
            pass

    return jsonify({
        'status':             'running',
        'started_at':         _started_at,
        'efs_base':           EFS_BASE,
        'csv_path':           CSV_PATH,
        'has_forecast':       os.path.exists(FORECAST_PATH),
        'has_sensor_data':    os.path.exists(CSV_PATH),
        'history_count':      len(glob.glob(f"{HISTORY_DIR}/forecast_*.json")),
        'forecast_info':      forecast_info,
    })


# ----------------------------------------------------------
# POST /api/upload_csv
# ----------------------------------------------------------
@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """
    Upload CSV — lưu vào local data/ và sao lưu sang EFS.
    Form-data: file=<csv_file>
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Missing "file" field in form-data'}), 400

    file = request.files['file']
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only .csv files are accepted'}), 400

    csv_text = file.read().decode('utf-8', errors='replace')
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    if not rows:
        return jsonify({'error': 'CSV file is empty or invalid'}), 400

    Path(CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        f.write(csv_text)
    print(f"[API] CSV saved → {CSV_PATH} ({len(rows)} rows)")

    return jsonify({
        'message':  'CSV đã lưu',
        'filename': file.filename,
        'path':     CSV_PATH,
        'num_rows': len(rows),
        'columns':  list(rows[0].keys()),
    })


# ----------------------------------------------------------
# GET /api/data/rows
# ----------------------------------------------------------
@app.route('/api/data/rows', methods=['GET'])
def get_data_rows():
    """Trả về toàn bộ last_48h.csv dạng JSON để app hiển thị editor."""
    if not os.path.exists(CSV_PATH):
        return jsonify({'error': 'No sensor data yet'}), 404
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return jsonify({'error': 'CSV is empty'}), 404
        return jsonify({'rows': rows, 'count': len(rows), 'source': CSV_PATH})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------
# POST /api/forecast/custom
# ----------------------------------------------------------
@app.route('/api/forecast/custom', methods=['POST'])
def custom_forecast():
    """
    Nhận rows đã chỉnh sửa từ app, lưu thành custom_YYYYMMDD_HHMMSS.csv,
    chạy lại inference, trả về forecast mới.

    Body JSON: { "rows": [ {timestamp, temperature, humidity, ...}, ... ] }
    """
    data = request.get_json(silent=True)
    if not data or 'rows' not in data:
        return jsonify({'error': 'Missing "rows" in request body'}), 400

    rows = data['rows']
    if not rows:
        return jsonify({'error': '"rows" array is empty'}), 400

    # --- Lưu custom CSV (không ghi đè last_48h.csv) ---
    Path(CUSTOM_DIR).mkdir(parents=True, exist_ok=True)
    ts_str      = datetime.now().strftime('%Y%m%d_%H%M%S')
    custom_path = f"{CUSTOM_DIR}/custom_{ts_str}.csv"

    fieldnames = list(rows[0].keys())
    with open(custom_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[API] Custom CSV saved → {custom_path} ({len(rows)} rows)")

    # --- Chạy inference với file custom ---
    if not _inference_lock.acquire(blocking=False):
        return jsonify({'error': 'Inference already running, try again shortly'}), 429

    custom_forecast_path = f"{CUSTOM_DIR}/forecast_custom_{ts_str}.json"
    try:
        cmd = [
            sys.executable, INFERENCE_SCRIPT,
            '--tcn',      TCN_MODEL,
            '--tcn_hard', TCN_HARD_MODEL,
            '--csv',      custom_path,
            '--output',   custom_forecast_path,
            '--device',   INFERENCE_DEVICE,
        ]
        if CLASSIFIER:
            cmd.extend(['--classifier', CLASSIFIER])
        if RAIN_PROB_CLS:
            cmd.extend(['--rain_prob_classifier', RAIN_PROB_CLS])

        print(f"[API] Running custom inference: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        if result.returncode != 0:
            print(f"[API] Inference stderr: {result.stderr[-400:]}")
            # Trả về forecast cũ nếu inference thất bại
            if os.path.exists(FORECAST_PATH):
                with open(FORECAST_PATH, 'r', encoding='utf-8') as f:
                    fallback = json.load(f)
                return jsonify({
                    'warning':     'Inference failed, returning last standard forecast',
                    'custom_file': os.path.basename(custom_path),
                    'forecast':    fallback.get('forecast', []),
                }), 200

            return jsonify({'error': f'Inference failed: {result.stderr[-200:]}'}), 500

        with open(custom_forecast_path, 'r', encoding='utf-8') as f:
            forecast_data = json.load(f)

        print(f"[API] Custom forecast done → {custom_forecast_path}")
        return jsonify({
            'message':     'Custom prediction complete',
            'custom_file': os.path.basename(custom_path),
            'num_rows':    len(rows),
            'forecast':    forecast_data.get('forecast', []),
        })

    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Inference timed out (>180s)'}), 504
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        _inference_lock.release()


# ============================================================
# Main
# ============================================================

def main():
    api_host = os.environ.get('API_HOST', '0.0.0.0')
    api_port = int(os.environ.get('API_PORT', 5000))

    print("=" * 60)
    print("  FORECAST API SERVER  —  AWS EFS Edition")
    print("=" * 60)
    print(f"  EFS Base   : {EFS_BASE}")
    print(f"  CSV        : {CSV_PATH}")
    print(f"  API        : http://{api_host}:{api_port}")
    print("=" * 60)
    print("[API] Endpoints:")
    print("       GET  /api/forecast")
    print("       GET  /api/forecast/latest")
    print("       GET  /api/forecast/step/<n>")
    print("       GET  /api/forecast/history")
    print("       GET  /api/forecast/history/<run_id>")
    print("       GET  /api/current")
    print("       GET  /api/status")
    print("       POST /api/upload_csv")
    print("       GET  /api/data/rows")
    print("       POST /api/forecast/custom\n")

    app.run(host=api_host, port=api_port, debug=False, threaded=True)


if __name__ == '__main__':
    main()