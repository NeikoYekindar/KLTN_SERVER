"""
=============================================================================
Forecast API Server  —  Flask + GCS
=============================================================================
Component 2: phục vụ Flask API, đọc dữ liệu từ Google Cloud Storage.
Không có MQTT, không chạy inference — chỉ là read-only API server.

Chạy:
    python server/api_server.py

Env vars (bắt buộc):
    GCS_BUCKET                      — tên GCS bucket
    GOOGLE_APPLICATION_CREDENTIALS  — path tới service account key JSON

Env vars (tùy chọn):
    GCS_REFRESH_INTERVAL  — giây giữa 2 lần refresh từ GCS (mặc định 300)
    API_HOST              — host Flask (mặc định 0.0.0.0)
    API_PORT              — port Flask (mặc định 5000)

Endpoints (giống server_dual_tcn.py):
    GET  /api/forecast
    GET  /api/forecast/latest
    GET  /api/forecast/step/<n>
    GET  /api/forecast/history
    GET  /api/forecast/history/<run_id>
    GET  /api/current
    GET  /api/status
=============================================================================
"""

import csv
import glob
import io
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

#import gcs_client

# ============================================================
# Config
# ============================================================

GCS_REFRESH_INTERVAL = int(os.environ.get('GCS_REFRESH_INTERVAL', 60))
GCS_FORECAST_PATH    = "forecasts/forecast_result.json"
GCS_CSV_PATH         = "data/last_48h.csv"
GCS_HISTORY_PREFIX   = "forecasts/history/"

# Local paths (cùng thư mục với worker.py)
LOCAL_CSV         = "data/last_48h.csv"
LOCAL_FORECAST    = "data/forecast_result.json"
LOCAL_HISTORY_DIR = "data/forecast_history"

# EFS paths (bản sao backup)
EFS_BASE         = os.environ.get('EFS_BASE', '/mnt/efs/fs1')
EFS_FORECAST     = f"{EFS_BASE}/forecasts/forecast_result.json"
EFS_CSV          = f"{EFS_BASE}/data/last_48h.csv"
EFS_HISTORY_DIR  = f"{EFS_BASE}/forecasts/history"

# ============================================================
# Global state (refreshed từ GCS định kỳ)
# ============================================================

latest_forecast       = None
latest_raw_data       = None
forecast_history_meta = []   # cache danh sách history (metadata nhẹ)

server_status = {
    'started_at':   None,
    'last_refresh': None,
    'errors':       [],
}


def _log_error(msg: str):
    print(f"[ERROR] {msg}")
    server_status['errors'].append({'time': datetime.now().isoformat(), 'error': msg})
    if len(server_status['errors']) > 20:
        server_status['errors'] = server_status['errors'][-20:]


# ============================================================
# GCS Refresh
# ============================================================

def _do_refresh():
    """Tải dữ liệu mới nhất từ EFS vào memory."""
    global latest_forecast, latest_raw_data, forecast_history_meta

    # 1. Forecast chính
    if os.path.exists(LOCAL_FORECAST):
        try:
            with open(LOCAL_FORECAST, 'r', encoding='utf-8') as f:
                data = json.load(f)
            latest_forecast = data
            print(f"[DATA] Refreshed forecast ({len(data.get('forecast', []))} steps)")
        except Exception as e:
            _log_error(f"Read forecast failed: {e}")

    # 2. Sensor CSV → latest_raw_data
    if os.path.exists(LOCAL_CSV):
        try:
            with open(LOCAL_CSV, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                latest_raw_data = {
                    'device_id':   'local',
                    'received_at': datetime.now().isoformat(),
                    'num_rows':    len(rows),
                    'last_row':    rows[-1],
                    'columns':     list(rows[0].keys()),
                }
                print(f"[DATA] Refreshed sensor data ({len(rows)} rows)")
        except Exception as e:
            _log_error(f"Read CSV failed: {e}")

    # 3. Forecast history metadata
    files = sorted(glob.glob(f"{LOCAL_HISTORY_DIR}/forecast_*.json"), reverse=True)[:50]
    history = []
    for path in files:
        filename = os.path.basename(path)
        stem   = filename.replace('.json', '')        # "forecast_20260415_120000"
        run_id = stem.replace('forecast_', '', 1)     # "20260415_120000"
        history.append({
            'run_id':   run_id,
            'filename': filename,
            'local_path': path,
        })
    if history:
        forecast_history_meta = history
        print(f"[DATA] Refreshed history list ({len(history)} entries)")

    server_status['last_refresh'] = datetime.now().isoformat()


def _refresh_loop():
    """Background thread: refresh EFS theo interval."""
    while True:
        try:
            _do_refresh()
        except Exception as e:
            _log_error(f"EFS refresh loop error: {e}")
        time.sleep(GCS_REFRESH_INTERVAL)


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
    if latest_forecast:
        return jsonify(latest_forecast)
    return jsonify({'error': 'No forecast available yet'}), 404


# ----------------------------------------------------------
# GET /api/forecast/latest
# ----------------------------------------------------------
@app.route('/api/forecast/latest', methods=['GET'])
def get_latest_step():
    if latest_forecast and latest_forecast.get('forecast'):
        return jsonify(latest_forecast['forecast'][0])
    return jsonify({'error': 'No forecast available'}), 404


# ----------------------------------------------------------
# GET /api/forecast/step/<n>
# ----------------------------------------------------------
@app.route('/api/forecast/step/<int:step>', methods=['GET'])
def get_forecast_step(step):
    if latest_forecast and latest_forecast.get('forecast'):
        for entry in latest_forecast['forecast']:
            if entry.get('horizon_step') == step:
                return jsonify(entry)
        return jsonify({'error': f'Step {step} not found (1–24)'}), 404
    return jsonify({'error': 'No forecast available'}), 404


# ----------------------------------------------------------
# GET /api/forecast/history
# ----------------------------------------------------------
@app.route('/api/forecast/history', methods=['GET'])
def get_forecast_history():
    limit = request.args.get('limit', 20, type=int)
    return jsonify({
        'count':   len(forecast_history_meta[:limit]),
        'history': forecast_history_meta[:limit],
    })


# ----------------------------------------------------------
# GET /api/forecast/history/<run_id>
# ----------------------------------------------------------
@app.route('/api/forecast/history/<run_id>', methods=['GET'])
def get_forecast_history_detail(run_id):
    path = f"{LOCAL_HISTORY_DIR}/forecast_{run_id}.json"
    if not os.path.exists(path):
        return jsonify({'error': f'run_id "{run_id}" not found'}), 404
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------
# GET /api/current
# ----------------------------------------------------------
@app.route('/api/current', methods=['GET'])
def get_current_data():
    if latest_raw_data:
        return jsonify(latest_raw_data)
    return jsonify({'error': 'No sensor data yet'}), 404


# ----------------------------------------------------------
# GET /api/status
# ----------------------------------------------------------
@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'status':                 'running',
        'efs_base':               EFS_BASE,
        'refresh_interval_secs':  GCS_REFRESH_INTERVAL,
        'has_forecast':           latest_forecast is not None,
        'has_sensor_data':        latest_raw_data is not None,
        'history_entries_cached': len(forecast_history_meta),
        **server_status,
    })

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """
    Upload file CSV — lưu vào thư mục local data/ và sao lưu sang EFS.
    Sau khi lưu, refresh dữ liệu in-memory ngay lập tức.

    Form-data: file=<csv_file>
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Thiếu field "file" trong form-data'}), 400

    file = request.files['file']
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Chỉ chấp nhận file .csv'}), 400

    csv_text = file.read().decode('utf-8', errors='replace')

    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    if not rows:
        return jsonify({'error': 'File CSV rỗng hoặc không hợp lệ'}), 400

    # Lưu local (primary)
    Path(LOCAL_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(LOCAL_CSV, 'w', encoding='utf-8', newline='') as f:
        f.write(csv_text)
    print(f"[API] CSV saved local: {file.filename} → {LOCAL_CSV} ({len(rows)} rows)")

    # Sao lưu sang EFS
    try:
        Path(EFS_CSV).parent.mkdir(parents=True, exist_ok=True)
        with open(EFS_CSV, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_text)
        print(f"[API] CSV backed up EFS: → {EFS_CSV}")
        efs_ok = True
    except Exception as e:
        print(f"[WARN] EFS backup failed: {e}")
        efs_ok = False

    threading.Thread(target=_do_refresh, daemon=True).start()

    return jsonify({
        'message':    'CSV đã lưu, đang refresh dữ liệu',
        'filename':   file.filename,
        'local_path': LOCAL_CSV,
        'efs_path':   EFS_CSV if efs_ok else None,
        'efs_ok':     efs_ok,
        'num_rows':   len(rows),
        'columns':    list(rows[0].keys()),
    })


# ============================================================
# Main
# ============================================================

def main():
    api_host = os.environ.get('API_HOST', '0.0.0.0')
    api_port = int(os.environ.get('API_PORT', 5000))

    print("=" * 60)
    print("  FORECAST API SERVER  —  AWS EFS Edition")
    print("=" * 60)
    print(f"  Local data    : {LOCAL_CSV}")
    print(f"  EFS backup    : {EFS_BASE}")
    print(f"  Refresh       : mỗi {GCS_REFRESH_INTERVAL}s")
    print(f"  API           : http://{api_host}:{api_port}")
    print("=" * 60)

    server_status['started_at'] = datetime.now().isoformat()

    print("[DATA] Initial load...")
    try:
        _do_refresh()
    except Exception as e:
        print(f"[WARN] Initial load failed: {e}")

    t = threading.Thread(target=_refresh_loop, daemon=True, name="data_refresh")
    t.start()

    print(f"\n[API] Running at http://{api_host}:{api_port}")
    print("[API] Endpoints:")
    print("       GET  /api/forecast")
    print("       GET  /api/forecast/latest")
    print("       GET  /api/forecast/step/<n>")
    print("       GET  /api/forecast/history")
    print("       GET  /api/forecast/history/<run_id>")
    print("       GET  /api/current")
    print("       GET  /api/status")
    print("       POST /api/upload_csv\n")

    app.run(host=api_host, port=api_port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
