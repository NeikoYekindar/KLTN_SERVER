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

Endpoints:
    GET  /api/forecast
    GET  /api/forecast/latest
    GET  /api/forecast/step/<n>
    GET  /api/forecast/history
    GET  /api/forecast/history/<run_id>
    GET  /api/current
    GET  /api/status
    POST /api/upload_csv
=============================================================================
"""

import csv
import glob
import io
import json
import os
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

# ============================================================
# Config
# ============================================================

# Local paths (cùng thư mục với worker.py, mounted vào /app/data)
LOCAL_CSV         = "data/last_48h.csv"
LOCAL_FORECAST    = "data/forecast_result.json"
LOCAL_HISTORY_DIR = "data/forecast_history"

# EFS paths (bản sao backup)
EFS_BASE         = os.environ.get('EFS_BASE', '/mnt/efs/fs1')
EFS_CSV          = f"{EFS_BASE}/data/last_48h.csv"
EFS_FORECAST     = f"{EFS_BASE}/forecasts/forecast_result.json"
EFS_HISTORY_DIR  = f"{EFS_BASE}/forecasts/history"

_started_at = datetime.now().isoformat()

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
    if not os.path.exists(LOCAL_FORECAST):
        return jsonify({'error': 'No forecast available yet'}), 404
    try:
        with open(LOCAL_FORECAST, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------
# GET /api/forecast/latest
# ----------------------------------------------------------
@app.route('/api/forecast/latest', methods=['GET'])
def get_latest_step():
    if not os.path.exists(LOCAL_FORECAST):
        return jsonify({'error': 'No forecast available'}), 404
    try:
        with open(LOCAL_FORECAST, 'r', encoding='utf-8') as f:
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
    if not os.path.exists(LOCAL_FORECAST):
        return jsonify({'error': 'No forecast available'}), 404
    try:
        with open(LOCAL_FORECAST, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for entry in data.get('forecast', []):
            if entry.get('horizon_step') == step:
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
    files = sorted(glob.glob(f"{LOCAL_HISTORY_DIR}/forecast_*.json"), reverse=True)[:limit]
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
    path = f"{LOCAL_HISTORY_DIR}/forecast_{run_id}.json"
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
    if not os.path.exists(LOCAL_CSV):
        return jsonify({'error': 'No sensor data yet'}), 404
    try:
        with open(LOCAL_CSV, 'r', encoding='utf-8') as f:
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
    return jsonify({
        'status':             'running',
        'started_at':         _started_at,
        'local_data':         LOCAL_CSV,
        'efs_base':           EFS_BASE,
        'has_forecast':       os.path.exists(LOCAL_FORECAST),
        'has_sensor_data':    os.path.exists(LOCAL_CSV),
        'history_count':      len(glob.glob(f"{LOCAL_HISTORY_DIR}/forecast_*.json")),
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

    # Lưu local (primary)
    Path(LOCAL_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(LOCAL_CSV, 'w', encoding='utf-8', newline='') as f:
        f.write(csv_text)
    print(f"[API] CSV saved local: {file.filename} → {LOCAL_CSV} ({len(rows)} rows)")

    # Sao lưu sang EFS
    efs_ok = False
    try:
        Path(EFS_CSV).parent.mkdir(parents=True, exist_ok=True)
        with open(EFS_CSV, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_text)
        print(f"[API] CSV backed up EFS → {EFS_CSV}")
        efs_ok = True
    except Exception as e:
        print(f"[WARN] EFS backup failed: {e}")

    return jsonify({
        'message':    'CSV đã lưu',
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
    print(f"  Local data : {LOCAL_CSV}")
    print(f"  EFS backup : {EFS_BASE}")
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
    print("       POST /api/upload_csv\n")

    app.run(host=api_host, port=api_port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
