"""
=============================================================================
Weather Forecast Server  —  Quick-run (no Docker)
=============================================================================
Gộp Worker + API Server vào một process duy nhất để chạy nhanh local.

  1. MQTT Subscriber  — nhận CSV từ Edge, lưu last.csv
  2. Inference        — gọi inference_dual_tcn.py (TCN + TCN-Hard + Classifiers)
  3. Flask API        — Unity / App GET kết quả

Usage:
    python server/server.py \
        --tcn      models/tcn_model.pth \
        --tcn_hard models/tcn_hard_model.pth \
        --classifier        models/condition_classifier.pkl \
        --rain_prob_classifier models/rain_prob_classifier.pkl

Env vars (optional):
    API_HOST, API_PORT
=============================================================================
"""

import argparse
import csv
import glob
import io
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    import paho.mqtt.client as mqtt
    MQTT_OK = True
except ImportError:
    MQTT_OK = False

# ============================================================
# Paths  (local, không dùng EFS)
# ============================================================

DATA_DIR             = Path("data")
MODELS_DIR           = Path("models")
CSV_PATH             = DATA_DIR / "last.csv"
FORECAST_PATH        = DATA_DIR / "forecast_result.json"
CSV_HISTORY_DIR      = DATA_DIR / "history"
FORECAST_HISTORY_DIR = DATA_DIR / "forecast_history"
CUSTOM_DIR           = DATA_DIR / "custom"

# ============================================================
# State
# ============================================================

inference_lock = threading.Lock()
_server_args   = None   # set in main()

server_status = {
    'started_at':          None,
    'last_data_received':  None,
    'last_inference_run':  None,
    'inference_count':     0,
    'last_inference_time': None,
    'errors':              [],
}


def _log_error(msg: str):
    print(f"[ERROR] {msg}")
    server_status['errors'].append({'time': datetime.now().isoformat(), 'error': msg})
    if len(server_status['errors']) > 20:
        server_status['errors'] = server_status['errors'][-20:]


def init_dirs():
    for d in [DATA_DIR, MODELS_DIR, CSV_HISTORY_DIR, FORECAST_HISTORY_DIR, CUSTOM_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def auto_detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[DEVICE] GPU: {name} → cuda")
            return 'cuda'
    except ImportError:
        pass
    print("[DEVICE] No GPU → cpu")
    return 'cpu'


# ============================================================
# Inference
# ============================================================

def run_inference(args, csv_path=None, output_path=None):
    """
    Chạy inference_dual_tcn.py qua subprocess.
    - csv_path / output_path: override (dùng cho custom forecast).
    Returns (success: bool, message: str).
    """
    global server_status

    is_custom        = csv_path is not None
    effective_csv    = Path(csv_path)    if is_custom else CSV_PATH
    effective_output = Path(output_path) if output_path else FORECAST_PATH

    if not effective_csv.exists():
        return False, "CSV file not found"

    if is_custom:
        acquired = inference_lock.acquire(blocking=True, timeout=200)
    else:
        acquired = inference_lock.acquire(blocking=False)

    if not acquired:
        print("[WARN] Inference already running, skipping")
        return False, "Inference already running"

    try:
        cmd = [
            sys.executable, args.inference_script,
            '--tcn',      args.tcn,
            '--tcn_hard', args.tcn_hard,
            '--csv',      str(effective_csv),
            '--output',   str(effective_output),
            '--device',   args.device,
        ]
        if args.classifier:
            cmd.extend(['--classifier', args.classifier])
        if args.rain_prob_classifier:
            cmd.extend(['--rain_prob_classifier', args.rain_prob_classifier])

        print(f"\n[INFERENCE] Running: {' '.join(cmd)}")
        t0 = time.time()
        env    = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding='utf-8', timeout=180, env=env)
        elapsed = time.time() - t0
        print(f"[INFERENCE] Done in {elapsed:.1f}s")

        if result.returncode == 0:
            print(result.stdout[-600:] if len(result.stdout) > 600 else result.stdout)

            if effective_output.exists() and not is_custom:
                ts_hist   = datetime.now().strftime('%Y%m%d_%H%M%S')
                hist_file = FORECAST_HISTORY_DIR / f"forecast_{ts_hist}.json"
                with open(effective_output, 'r', encoding='utf-8') as f:
                    forecast_data = json.load(f)
                hist_file.write_text(
                    json.dumps({'run_id': ts_hist, 'saved_at': datetime.now().isoformat(),
                                **forecast_data}, ensure_ascii=False, indent=2),
                    encoding='utf-8'
                )
                server_status['last_inference_run']  = datetime.now().isoformat()
                server_status['last_inference_time'] = round(elapsed, 2)
                server_status['inference_count']    += 1

            return True, str(effective_output)

        else:
            msg = f"Inference failed (rc={result.returncode}): {result.stderr[-400:]}"
            _log_error(msg)
            return False, msg

    except subprocess.TimeoutExpired:
        _log_error("Inference timeout (>180s)")
        return False, "Timeout"
    except Exception as e:
        _log_error(f"run_inference error: {e}")
        return False, str(e)
    finally:
        inference_lock.release()


# ============================================================
# MQTT
# ============================================================

def process_incoming_data(payload_str: str, args):
    try:
        payload   = json.loads(payload_str)
        csv_data  = payload.get('csv_data', '')
        device_id = payload.get('device_id', 'unknown')
        num_rows  = payload.get('num_rows', 0)

        print(f"\n[DATA] Received {num_rows} rows from {device_id}")

        if not csv_data:
            _log_error("Payload missing csv_data")
            return

        CSV_PATH.write_text(csv_data, encoding='utf-8')
        print(f"[DATA] Saved → {CSV_PATH}")

        ts           = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_file = CSV_HISTORY_DIR / f"data_{ts}_{device_id}.csv"
        history_file.write_text(csv_data, encoding='utf-8')

        server_status['last_data_received'] = datetime.now().isoformat()

        run_inference(args)

    except json.JSONDecodeError as e:
        _log_error(f"JSON parse error: {e}")
    except Exception as e:
        _log_error(f"process_incoming_data error: {e}")


def start_mqtt_subscriber(args):
    if not MQTT_OK:
        print("[WARN] paho-mqtt not installed — MQTT disabled")
        return None

    def on_connect(client, _userdata, _flags, rc):
        if rc == 0:
            client.subscribe(args.topic, qos=1)
            print(f"[MQTT] Subscribed → '{args.topic}'")
        else:
            print(f"[MQTT] Connection failed: rc={rc}")

    def on_disconnect(_client, _userdata, rc):
        if rc != 0:
            print(f"[MQTT] Disconnected (rc={rc}), reconnecting...")

    def on_message(_client, _userdata, msg):
        print(f"[MQTT] Message topic='{msg.topic}' size={len(msg.payload)}B")
        threading.Thread(
            target=process_incoming_data,
            args=(msg.payload.decode('utf-8'), args),
            daemon=True,
        ).start()

    client = mqtt.Client(client_id=f"server_{os.getpid()}", protocol=mqtt.MQTTv311)
    if args.mqtt_username:
        client.username_pw_set(args.mqtt_username, args.mqtt_password)
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_message    = on_message
    client.reconnect_delay_set(min_delay=2, max_delay=60)

    try:
        client.connect(args.broker, args.mqtt_port, keepalive=120)
        client.loop_start()
        print(f"[MQTT] Connected to {args.broker}:{args.mqtt_port}")
        return client
    except Exception as e:
        print(f"[MQTT] Cannot connect: {e}")
        return None


# ============================================================
# Flask API
# ============================================================

app = Flask(__name__)
CORS(app)


@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    if not FORECAST_PATH.exists():
        return jsonify({'error': 'No forecast available yet'}), 404
    try:
        with open(FORECAST_PATH, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast/latest', methods=['GET'])
def get_latest_step():
    if not FORECAST_PATH.exists():
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


@app.route('/api/forecast/step/<int:step>', methods=['GET'])
def get_forecast_step(step):
    if not FORECAST_PATH.exists():
        return jsonify({'error': 'No forecast available'}), 404
    try:
        with open(FORECAST_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for entry in data.get('forecast', []):
            if entry.get('horizon_step') == step:
                rp = entry.get('rain_probability')
                if rp is not None and rp not in (0, 45, 100, 0.0, 45.0, 100.0):
                    entry['rain_probability_raw'] = rp
                    entry['rain_probability'] = min([0, 45, 100], key=lambda v: abs(v - rp))
                return jsonify(entry)
        return jsonify({'error': f'Step {step} not found (1–24)'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast/history', methods=['GET'])
def get_forecast_history():
    limit = request.args.get('limit', 20, type=int)
    files = sorted(glob.glob(str(FORECAST_HISTORY_DIR / 'forecast_*.json')), reverse=True)[:limit]
    history = []
    for path in files:
        filename = os.path.basename(path)
        run_id   = filename.replace('.json', '').replace('forecast_', '', 1)
        history.append({'run_id': run_id, 'filename': filename})
    return jsonify({'count': len(history), 'history': history})


@app.route('/api/forecast/history/<run_id>', methods=['GET'])
def get_forecast_history_detail(run_id):
    path = FORECAST_HISTORY_DIR / f"forecast_{run_id}.json"
    if not path.exists():
        return jsonify({'error': f'run_id "{run_id}" not found'}), 404
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/current', methods=['GET'])
def get_current_data():
    if not CSV_PATH.exists():
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


@app.route('/api/status', methods=['GET'])
def get_status():
    forecast_info = {}
    if FORECAST_PATH.exists():
        try:
            with open(FORECAST_PATH, 'r', encoding='utf-8') as f:
                fd = json.load(f)
            steps = fd.get('forecast', [])
            forecast_info = {
                'steps':                len(steps),
                'has_rain_probability': any('rain_probability' in s for s in steps),
                'has_uv_index':         any('uv_index' in s for s in steps),
                'generated_at':         fd.get('generated_at'),
            }
        except Exception:
            pass

    return jsonify({
        'status':          'running',
        'has_forecast':    FORECAST_PATH.exists(),
        'has_sensor_data': CSV_PATH.exists(),
        'history_count':   len(glob.glob(str(FORECAST_HISTORY_DIR / 'forecast_*.json'))),
        'forecast_info':   forecast_info,
        **server_status,
    })


@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'Missing "file" field in form-data'}), 400

    file = request.files['file']
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only .csv files are accepted'}), 400

    csv_text = file.read().decode('utf-8', errors='replace')
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    if not rows:
        return jsonify({'error': 'CSV file is empty or invalid'}), 400

    CSV_PATH.write_text(csv_text, encoding='utf-8')
    print(f"[API] CSV saved → {CSV_PATH} ({len(rows)} rows)")

    threading.Thread(
        target=run_inference,
        args=(_server_args,),
        daemon=True,
    ).start()

    return jsonify({
        'message':  'CSV đã lưu, inference đang chạy',
        'filename': file.filename,
        'num_rows': len(rows),
        'columns':  list(rows[0].keys()),
    })


@app.route('/api/data/rows', methods=['GET'])
def get_data_rows():
    if not CSV_PATH.exists():
        return jsonify({'error': 'No sensor data yet'}), 404
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return jsonify({'error': 'CSV is empty'}), 404
        return jsonify({'rows': rows, 'count': len(rows), 'source': str(CSV_PATH)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast/custom', methods=['POST'])
def custom_forecast():
    data = request.get_json(silent=True)
    if not data or 'rows' not in data:
        return jsonify({'error': 'Missing "rows" in request body'}), 400

    rows = data['rows']
    if not rows:
        return jsonify({'error': '"rows" array is empty'}), 400

    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    ts_str               = datetime.now().strftime('%Y%m%d_%H%M%S')
    custom_csv_path      = CUSTOM_DIR / f"custom_{ts_str}.csv"
    custom_forecast_path = CUSTOM_DIR / f"forecast_custom_{ts_str}.json"

    fieldnames = list(rows[0].keys())
    with open(custom_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[API] Custom CSV saved → {custom_csv_path} ({len(rows)} rows)")

    ok, msg = run_inference(
        _server_args,
        csv_path=str(custom_csv_path),
        output_path=str(custom_forecast_path),
    )

    if ok and custom_forecast_path.exists():
        with open(custom_forecast_path, 'r', encoding='utf-8') as f:
            fd = json.load(f)
        return jsonify({
            'message':     'Custom prediction complete',
            'custom_file': custom_csv_path.name,
            'num_rows':    len(rows),
            'forecast':    fd.get('forecast', []),
        })

    # Fallback: trả forecast chuẩn nếu inference thất bại
    if FORECAST_PATH.exists():
        with open(FORECAST_PATH, 'r', encoding='utf-8') as f:
            fallback = json.load(f)
        return jsonify({
            'warning':  f'Inference failed, returning last standard forecast: {msg}',
            'forecast': fallback.get('forecast', []),
        })

    return jsonify({'error': f'Inference failed: {msg}'}), 500


@app.route('/api/trigger', methods=['POST'])
def trigger_inference():
    if not CSV_PATH.exists():
        return jsonify({'error': 'No CSV data available'}), 400
    threading.Thread(target=run_inference, args=(_server_args,), daemon=True).start()
    return jsonify({'message': 'Inference triggered'})


# ============================================================
# Main
# ============================================================

def main():
    global _server_args

    parser = argparse.ArgumentParser(description="Weather Forecast Server — Quick-run")

    # MQTT
    parser.add_argument('--broker',           default='localhost')
    parser.add_argument('--mqtt-port',        type=int, default=1883)
    parser.add_argument('--topic',            default='weather/data')
    parser.add_argument('--mqtt-username',    default=None)
    parser.add_argument('--mqtt-password',    default=None)

    # Inference
    parser.add_argument('--inference-script', default='server/inference_dual_tcn.py')
    parser.add_argument('--tcn',              required=True,  help='TCNForecaster .pth')
    parser.add_argument('--tcn_hard',         required=True,  help='HardTargetForecaster .pth')
    parser.add_argument('--classifier',       default=None,   help='condition_classifier.pkl')
    parser.add_argument('--rain_prob_classifier', default=None, help='rain_prob_classifier.pkl')
    parser.add_argument('--device',           default=None)

    # API
    parser.add_argument('--api-host', default=os.environ.get('API_HOST', '0.0.0.0'))
    parser.add_argument('--api-port', type=int, default=int(os.environ.get('API_PORT', 5000)))

    args = parser.parse_args()

    if args.device is None:
        args.device = auto_detect_device()

    for attr, flag in [('tcn', '--tcn'), ('tcn_hard', '--tcn_hard')]:
        if not os.path.exists(getattr(args, attr)):
            parser.error(f"{flag}: file not found")

    if args.classifier and not os.path.exists(args.classifier):
        parser.error(f"--classifier: file not found '{args.classifier}'")

    if args.rain_prob_classifier and not os.path.exists(args.rain_prob_classifier):
        parser.error(f"--rain_prob_classifier: file not found '{args.rain_prob_classifier}'")

    if not os.path.exists(args.inference_script):
        parser.error(f"--inference-script: not found '{args.inference_script}'")

    init_dirs()

    _server_args = args
    server_status['started_at'] = datetime.now().isoformat()

    print("=" * 60)
    print("  WEATHER FORECAST SERVER  —  Quick-run (Dual TCN)")
    print("=" * 60)
    print(f"  MQTT            : {args.broker}:{args.mqtt_port}")
    print(f"  Topic           : {args.topic}")
    print(f"  TCN             : {args.tcn}")
    print(f"  TCN-Hard        : {args.tcn_hard}")
    print(f"  Classifier      : {args.classifier or 'N/A'}")
    print(f"  Rain Prob Class : {args.rain_prob_classifier or 'N/A'}")
    print(f"  Device          : {args.device}")
    print(f"  API             : http://{args.api_host}:{args.api_port}")
    print("=" * 60)
    print("[API] Endpoints:")
    print("       GET  /api/forecast")
    print("       GET  /api/forecast/latest")
    print("       GET  /api/forecast/step/<n>")
    print("       GET  /api/forecast/history")
    print("       GET  /api/forecast/history/<run_id>")
    print("       GET  /api/current")
    print("       GET  /api/status")
    print("       GET  /api/data/rows")
    print("       POST /api/upload_csv")
    print("       POST /api/forecast/custom")
    print("       POST /api/trigger\n")

    mqtt_client = start_mqtt_subscriber(args)

    try:
        app.run(host=args.api_host, port=args.api_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        print("\n[INFO] Server stopped.")


if __name__ == '__main__':
    main()
