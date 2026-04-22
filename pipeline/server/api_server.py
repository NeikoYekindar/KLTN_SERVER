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
import io
import json
import os
import threading
import time
from datetime import datetime

from flask import Flask, jsonify, request
from flask_cors import CORS

import gcs_client

# ============================================================
# Config
# ============================================================

GCS_REFRESH_INTERVAL = int(os.environ.get('GCS_REFRESH_INTERVAL', 300))
GCS_FORECAST_PATH    = "forecasts/forecast_result.json"
GCS_CSV_PATH         = "data/last_48h.csv"
GCS_HISTORY_PREFIX   = "forecasts/history/"

# ============================================================
# Global state (refreshed từ GCS định kỳ)
# ============================================================

latest_forecast       = None
latest_raw_data       = None
forecast_history_meta = []   # cache danh sách history (metadata nhẹ)

server_status = {
    'started_at':      None,
    'last_gcs_refresh': None,
    'errors':          [],
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
    """Tải dữ liệu mới nhất từ GCS vào memory."""
    global latest_forecast, latest_raw_data, forecast_history_meta

    # 1. Forecast chính
    data = gcs_client.download_json(GCS_FORECAST_PATH)
    if data:
        latest_forecast = data
        print(f"[GCS] Refreshed forecast ({len(data.get('forecast', []))} steps)")

    # 2. Sensor CSV → latest_raw_data
    csv_text = gcs_client.download_text(GCS_CSV_PATH)
    if csv_text:
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        if rows:
            latest_raw_data = {
                'device_id':   'gcs',
                'received_at': datetime.now().isoformat(),
                'num_rows':    len(rows),
                'last_row':    rows[-1],
                'columns':     list(rows[0].keys()),
            }
            print(f"[GCS] Refreshed sensor data ({len(rows)} rows)")

    # 3. Forecast history metadata (tên blob → metadata nhẹ)
    blob_names = gcs_client.list_blobs(GCS_HISTORY_PREFIX)
    history = []
    for name in sorted(blob_names, reverse=True)[:50]:   # 50 lần gần nhất
        # Tên blob: "forecasts/history/forecast_20260415_120000.json"
        stem = name.split('/')[-1].replace('.json', '')   # "forecast_20260415_120000"
        run_id = stem.replace('forecast_', '', 1)         # "20260415_120000"
        history.append({
            'run_id':   run_id,
            'filename': name.split('/')[-1],
            'gcs_path': name,
        })
    if history:
        forecast_history_meta = history
        print(f"[GCS] Refreshed history list ({len(history)} entries)")

    server_status['last_gcs_refresh'] = datetime.now().isoformat()


def _refresh_loop():
    """Background thread: refresh GCS theo interval."""
    while True:
        try:
            _do_refresh()
        except Exception as e:
            _log_error(f"GCS refresh loop error: {e}")
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
    gcs_path = f"forecasts/history/forecast_{run_id}.json"
    data = gcs_client.download_json(gcs_path)
    if data is None:
        return jsonify({'error': f'run_id "{run_id}" not found'}), 404
    return jsonify(data)


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
        'status':                'running',
        'gcs_bucket':            os.environ.get('GCS_BUCKET', ''),
        'refresh_interval_secs': GCS_REFRESH_INTERVAL,
        'has_forecast':          latest_forecast is not None,
        'has_sensor_data':       latest_raw_data is not None,
        'history_entries_cached': len(forecast_history_meta),
        **server_status,
    })


# ============================================================
# Main
# ============================================================

def main():
    api_host = os.environ.get('API_HOST', '0.0.0.0')
    api_port = int(os.environ.get('API_PORT', 5000))
    bucket   = os.environ.get('GCS_BUCKET', '')

    print("=" * 60)
    print("  FORECAST API SERVER  —  GCS Edition")
    print("=" * 60)
    print(f"  GCS Bucket    : {bucket or 'NOT SET !'}")
    print(f"  Refresh       : mỗi {GCS_REFRESH_INTERVAL}s")
    print(f"  API           : http://{api_host}:{api_port}")
    print("=" * 60)

    server_status['started_at'] = datetime.now().isoformat()

    # Lần đầu load ngay từ GCS (đồng bộ)
    print("[GCS] Initial load...")
    try:
        _do_refresh()
    except Exception as e:
        print(f"[WARN] Initial GCS load failed: {e}")

    # Background refresh thread
    t = threading.Thread(target=_refresh_loop, daemon=True, name="gcs_refresh")
    t.start()

    print(f"\n[API] Running at http://{api_host}:{api_port}")
    print("[API] Endpoints:")
    print("       GET  /api/forecast")
    print("       GET  /api/forecast/latest")
    print("       GET  /api/forecast/step/<n>")
    print("       GET  /api/forecast/history")
    print("       GET  /api/forecast/history/<run_id>")
    print("       GET  /api/current")
    print("       GET  /api/status\n")

    app.run(host=api_host, port=api_port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
