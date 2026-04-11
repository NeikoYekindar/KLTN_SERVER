"""
=============================================================================
Weather Forecast Server
=============================================================================
Runs on Cloud/Server:
  1. MQTT Subscriber — receives 48h CSV from Edge
  2. Saves CSV → last_48h.csv
  3. Runs inference.py → forecast_result.json
  4. Flask API — Unity GET results

Usage:
    python server.py --broker localhost --tcn models/tcn_model.pth \
                     --arima models/arima_model.pkl \
                     --ensemble_json models/ensemble_metrics.json
=============================================================================
"""

import argparse
import csv
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
# Config
# ============================================================

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
CSV_PATH = DATA_DIR / "last_48h.csv"
FORECAST_PATH = DATA_DIR / "forecast_result.json"
HISTORY_DIR = DATA_DIR / "history"

FORECAST_TCN_PATH = DATA_DIR / "forecast_tcn.json"
FORECAST_LSTM_PATH = DATA_DIR / "forecast_lstm.json"

# Global state
latest_forecast_tcn = None
latest_forecast_lstm = None

latest_forecast = None
latest_raw_data = None
server_status = {
    'started_at': None,
    'last_data_received': None,
    'last_inference_tcn': None,
    'last_inference_lstm': None,
    'inference_count_tcn': 0,
    'inference_count_lstm': 0,
    'errors': [],
}


def init_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    HISTORY_DIR.mkdir(exist_ok=True)


# ============================================================
# Parse CSV and update latest_raw_data
# ============================================================

def update_raw_data_from_csv(csv_path, device_id='unknown'):
    """Parse CSV file → update latest_raw_data for /api/current."""
    global latest_raw_data, server_status

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if rows:
            latest_raw_data = {
                'device_id': device_id,
                'received_at': datetime.now().isoformat(),
                'num_rows': len(rows),
                'last_row': rows[-1],
                'columns': list(rows[0].keys()),
            }
            server_status['last_data_received'] = datetime.now().isoformat()
            print(f"[DATA] Updated current: {len(rows)} rows from {device_id}")
    except Exception as e:
        print(f"[WARN] CSV parse error: {e}")


# ============================================================
# Process incoming data from MQTT
# ============================================================

def process_incoming_data(payload_str, args):
    """
    Process JSON payload from MQTT:
    - Parse CSV data from edge
    - Save file last_48h.csv
    - Save history copy
    - Run inference
    """
    try:
        payload = json.loads(payload_str)
        csv_data = payload.get('csv_data', '')
        device_id = payload.get('device_id', 'unknown')
        num_rows = payload.get('num_rows', 0)

        print(f"\n[MQTT] Received {num_rows} rows from {device_id}")

        if not csv_data:
            print("[ERROR] Payload has no csv_data")
            return

        # Save main CSV
        with open(CSV_PATH, 'w') as f:
            f.write(csv_data)
        print(f"[DATA] Saved → {CSV_PATH}")

        # Save history copy
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_file = HISTORY_DIR / f"data_{ts}.csv"
        with open(history_file, 'w') as f:
            f.write(csv_data)

        # Update /api/current
        update_raw_data_from_csv(CSV_PATH, device_id)

        # Run inference
        #run_inference(args)
        run_both_inference(args)

    except json.JSONDecodeError as e:
        err = f"JSON parse error: {e}"
        print(f"[ERROR] {err}")
        server_status['errors'].append({'time': datetime.now().isoformat(), 'error': err})
    except Exception as e:
        err = f"Process error: {e}"
        print(f"[ERROR] {err}")
        server_status['errors'].append({'time': datetime.now().isoformat(), 'error': err})


# ============================================================
# Run inference.py
# ============================================================
def run_both_inference(args):
    """Run both TCN+ARIMA and LSTM+ARIMA in parallel."""
    threads = []

    if args.tcn:
        t = threading.Thread(target=run_inference_tcn, args=(args,), name="inference_tcn")
        threads.append(t)
        
    if args.lstm:
        t = threading.Thread(target=run_inference_lstm, args=(args,), name="inference_lstm")
        threads.append(t)
    
    for t in threads:
        t.start()
    
    # Don't wait — keep non-blocking




def run_inference_tcn(args):
    global latest_forecast_tcn, server_status
    
    if not CSV_PATH.exists():
        return
    
    cmd = [sys.executable, args.inference_script_tcn,
           '--csv', str(CSV_PATH),
           '--output', str(FORECAST_TCN_PATH),
           '--device', args.device]
    
    if args.tcn:
        cmd.extend(['--tcn', args.tcn])
    if args.arima:
        cmd.extend(['--arima', args.arima])
    if args.ensemble_json_tcn:
        cmd.extend(['--ensemble_json', args.ensemble_json_tcn])
    
    print(f"\n[TCN] Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - start_time

        print(f"[TCN] Completed in {elapsed:.1f}s")

        if result.returncode == 0:
            if FORECAST_TCN_PATH.exists():
                with open(FORECAST_TCN_PATH, 'r') as f:
                    latest_forecast_tcn = json.load(f)
                print(f"[TCN] Loaded: {len(latest_forecast_tcn.get('forecast', []))} steps")
            server_status['last_inference_tcn'] = datetime.now().isoformat()
            server_status['inference_count_tcn'] += 1
        else:
            err = f"TCN failed (rc={result.returncode}): {result.stderr[-300:]}"
            print(f"[ERROR] {err}")
            server_status['errors'].append({'time': datetime.now().isoformat(), 'error': err})
    
    except subprocess.TimeoutExpired:
        print("[ERROR] TCN inference timeout (>120s)")
    except Exception as e:
        print(f"[ERROR] TCN: {e}")


def run_inference_lstm(args):
    global latest_forecast_lstm, server_status
    
    if not CSV_PATH.exists():
        return
    
    
    cmd = [sys.executable, args.inference_script_lstm,
           '--csv', str(CSV_PATH),
           '--output', str(FORECAST_LSTM_PATH),
           '--device', args.device]

    if args.lstm:
        cmd.extend(['--lstm', args.lstm])
    if args.arima:
        cmd.extend(['--arima', args.arima])
    if args.ensemble_json_lstm:
        cmd.extend(['--ensemble_json', args.ensemble_json_lstm])

    print(f"\n[LSTM] Running: {' '.join(cmd)}")
    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - start_time
        print(f"[LSTM] Completed in {elapsed:.1f}s")

        if result.returncode == 0:
            if FORECAST_LSTM_PATH.exists():
                with open(FORECAST_LSTM_PATH, 'r') as f:
                    latest_forecast_lstm = json.load(f)
                print(f"[LSTM] Loaded: {len(latest_forecast_lstm.get('forecast', []))} steps")
            server_status['last_inference_lstm'] = datetime.now().isoformat()
            server_status['inference_count_lstm'] += 1
        else:
            err = f"LSTM failed (rc={result.returncode}): {result.stderr[-300:]}"
            print(f"[ERROR] {err}")
            server_status['errors'].append({'time': datetime.now().isoformat(), 'error': err})

    except subprocess.TimeoutExpired:
        print("[ERROR] LSTM inference timeout (>120s)")
    except Exception as e:
        print(f"[ERROR] LSTM: {e}")


def run_inference(args):
    """Call inference.py subprocess."""
    global latest_forecast, server_status

    if not CSV_PATH.exists():
        print("[WARN] No CSV file found, skipping inference")
        return

    # Build command
    cmd = [sys.executable, args.inference_script, '--csv', str(CSV_PATH),
           '--output', str(FORECAST_PATH), '--device', args.device]

    if args.tcn:
        cmd.extend(['--tcn', args.tcn])
    if args.arima:
        cmd.extend(['--arima', args.arima])
    if args.ensemble_json:
        cmd.extend(['--ensemble_json', args.ensemble_json])
    if args.w_tcn:
        cmd.extend(['--w_tcn', str(args.w_tcn)])
    if args.w_arima:
        cmd.extend(['--w_arima', str(args.w_arima)])

    print(f"\n[INFERENCE] Running: {' '.join(cmd)}")
    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - start_time
        print(f"[INFERENCE] Completed in {elapsed:.1f}s")

        if result.returncode == 0:
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

            if FORECAST_PATH.exists():
                with open(FORECAST_PATH, 'r') as f:
                    latest_forecast = json.load(f)
                print(f"[INFERENCE] Loaded forecast: {len(latest_forecast.get('forecast', []))} steps")

            server_status['last_inference_run'] = datetime.now().isoformat()
            server_status['inference_count'] += 1
        else:
            err = f"Inference failed (rc={result.returncode}): {result.stderr[-300:]}"
            print(f"[ERROR] {err}")
            server_status['errors'].append({'time': datetime.now().isoformat(), 'error': err})

    except subprocess.TimeoutExpired:
        err = "Inference timeout (>120s)"
        print(f"[ERROR] {err}")
        server_status['errors'].append({'time': datetime.now().isoformat(), 'error': err})
    except Exception as e:
        err = f"Inference error: {e}"
        print(f"[ERROR] {err}")
        server_status['errors'].append({'time': datetime.now().isoformat(), 'error': err})


# ============================================================
# MQTT Subscriber
# ============================================================

def start_mqtt_subscriber(args):
    """Start MQTT subscriber thread."""
    if not MQTT_OK:
        print("[WARN] paho-mqtt not installed, skipping MQTT subscriber")
        return None

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(args.topic, qos=1)
            print(f"[MQTT] Subscribed → '{args.topic}'")
        else:
            print(f"[MQTT] Connection failed: rc={rc}")

    def on_message(client, userdata, msg):
        print(f"[MQTT] Received message from topic '{msg.topic}' ({len(msg.payload)} bytes)")
        thread = threading.Thread(
            target=process_incoming_data,
            args=(msg.payload.decode('utf-8'), args)
        )
        thread.start()

    client = mqtt.Client(client_id=f"server_{os.getpid()}", protocol=mqtt.MQTTv311)

    if args.mqtt_username:
        client.username_pw_set(args.mqtt_username, args.mqtt_password)

    client.on_connect = on_connect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=60)

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


@app.route('/api/tcn/forecast', methods=['GET'])
def get_forecast_tcn():
    """Full 6h forecast."""
    if latest_forecast_tcn:
        return jsonify(latest_forecast_tcn)
    if FORECAST_PATH.exists():
        with open(FORECAST_PATH, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'No forecast available yet'}), 404


@app.route('/api/tcn/forecast/latest', methods=['GET'])
def get_latest__tcn_step():
    """Only step h+1."""
    if latest_forecast_tcn and latest_forecast_tcn.get('forecast'):
        return jsonify(latest_forecast_tcn['forecast'][0])
    return jsonify({'error': 'No forecast available'}), 404


@app.route('/api/tcn/forecast/step/<int:step>', methods=['GET'])
def get_forecast_tcn_step(step):
    """Specific step (1-6)."""
    if latest_forecast_tcn and latest_forecast_tcn.get('forecast'):
        for entry in latest_forecast_tcn['forecast']:
            if entry.get('horizon_step') == step:
                return jsonify(entry)
    return jsonify({'error': f'Step {step} not found'}), 404


@app.route('/api/lstm/forecast', methods=['GET'])
def get_forecast_lstm():
    """Full LSTM+ARIMA 6h forecast."""
    if latest_forecast_lstm:
        return jsonify(latest_forecast_lstm)
    if FORECAST_LSTM_PATH.exists():
        with open(FORECAST_LSTM_PATH, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'No LSTM forecast available'}), 404

@app.route('/api/lstm/forecast/latest', methods=['GET'])
def get_forecast_lstm_latest():
    """LSTM: only step h+1."""
    if latest_forecast_lstm and latest_forecast_lstm.get('forecast'):
        return jsonify(latest_forecast_lstm['forecast'][0])
    return jsonify({'error': 'No LSTM forecast available'}), 404


@app.route('/api/lstm/forecast/step/<int:step>', methods=['GET'])
def get_forecast_lstm_step(step):
    """LSTM: specific step (1-6)."""
    if latest_forecast_lstm and latest_forecast_lstm.get('forecast'):
        for entry in latest_forecast_lstm['forecast']:
            if entry.get('horizon_step') == step:
                return jsonify(entry)
    return jsonify({'error': f'LSTM step {step} not found'}), 404



@app.route('/api/compare', methods=['GET'])
def compare_models():
    """Compare TCN vs LSTM results side by side."""
    result = {
        'tcn': latest_forecast_tcn if latest_forecast_tcn else None,
        'lstm': latest_forecast_lstm if latest_forecast_lstm else None,
    }

    # Calculate difference if both are available
    if latest_forecast_tcn and latest_forecast_lstm:
        tcn_fc = latest_forecast_tcn.get('forecast', [])
        lstm_fc = latest_forecast_lstm.get('forecast', [])

        if tcn_fc and lstm_fc:
            diff = []
            for t_step, l_step in zip(tcn_fc, lstm_fc):
                d = {'horizon_step': t_step.get('horizon_step')}
                for key in t_step:
                    if key in ('timestamp', 'horizon_step'):
                        continue
                    if key in l_step:
                        t_val = t_step[key]
                        l_val = l_step[key]
                        d[key] = {
                            'tcn': round(t_val, 4),
                            'lstm': round(l_val, 4),
                            'diff': round(abs(t_val - l_val), 4),
                        }
                diff.append(d)
            result['comparison'] = diff

    return jsonify(result)




@app.route('/api/forecast', methods=['GET'])
def get_forecast_default():
    """Backward compatible — returns TCN by default."""
    return get_forecast_tcn()


@app.route('/api/current', methods=['GET'])
def get_current_data():
    """Latest sensor data."""
    if latest_raw_data:
        return jsonify(latest_raw_data)
    return jsonify({'error': 'No sensor data yet'}), 404


@app.route('/api/status', methods=['GET'])
def get_status():
    """Health check."""
    return jsonify({
        'status': 'running',
        **server_status,
        'has_forecast_tcn': latest_forecast_tcn is not None,
        'has_forecast_lstm': latest_forecast_lstm is not None,
        'has_sensor_data': latest_raw_data is not None,
    })


@app.route('/api/trigger', methods=['POST'])
def trigger_inference():
    """Manually trigger both inferences."""
    if CSV_PATH.exists():
        threading.Thread(target=run_both_inference, args=(app.config['args'],)).start()
        return jsonify({'message': 'Both TCN and LSTM inference triggered'})
    return jsonify({'error': 'No CSV data available'}), 400


@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """Upload CSV directly (replaces MQTT for testing)."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file.save(str(CSV_PATH))
    print(f"[API] Received CSV upload: {file.filename}")

    # Update /api/current
    update_raw_data_from_csv(CSV_PATH, device_id='upload')

    # Trigger inference
    thread = threading.Thread(target=run_both_inference, args=(app.config['args'],))
    thread.start()

    return jsonify({'message': 'CSV saved, inference triggered'})


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Weather Forecast Server")

    # MQTT
    parser.add_argument('--broker', type=str, default='localhost')
    parser.add_argument('--mqtt-port', type=int, default=1883)
    parser.add_argument('--topic', type=str, default='weather/data')
    parser.add_argument('--mqtt-username', type=str, default=None)
    parser.add_argument('--mqtt-password', type=str, default=None)

    # TCN + ARIMA
    parser.add_argument('--tcn', type=str, default=None, help='TCN model (.pth)')
    parser.add_argument('--inference-script-tcn', type=str, default='server/inference.py')
    parser.add_argument('--ensemble_json_tcn', type=str, default=None,
                        help='Ensemble weights JSON for TCN+ARIMA')

    # LSTM + ARIMA
    parser.add_argument('--lstm', type=str, default=None, help='LSTM model (.pth)')
    parser.add_argument('--inference-script-lstm', type=str, default='server/inference_lstm_arima.py')
    parser.add_argument('--ensemble_json_lstm', type=str, default=None,
                        help='Ensemble weights JSON for LSTM+ARIMA')

    # ARIMA (shared for both)
    parser.add_argument('--arima', type=str, default=None, help='ARIMA model (.pkl)')

    # General
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--api-port', type=int, default=5000)
    parser.add_argument('--api-host', type=str, default='0.0.0.0')

    args = parser.parse_args()

    if not args.tcn and not args.arima:
        parser.error("At least --tcn or --arima is required")

    init_dirs()

    print("=" * 60)
    print("  WEATHER FORECAST SERVER v2 — DUAL MODEL")
    print("=" * 60)
    print(f"  MQTT       : {args.broker}:{args.mqtt_port}")
    print(f"  Topic      : {args.topic}")
    print(f"  TCN Model  : {args.tcn or 'N/A'}")
    print(f"  LSTM Model : {args.lstm or 'N/A'}")
    print(f"  ARIMA      : {args.arima or 'N/A'}")
    print(f"  Ensemble TCN  : {args.ensemble_json_tcn or 'N/A'}")
    print(f"  Ensemble LSTM : {args.ensemble_json_lstm or 'N/A'}")
    print(f"  API        : http://{args.api_host}:{args.api_port}")
    print("=" * 60)

    server_status['started_at'] = datetime.now().isoformat()

    # Load existing forecast if available
    global latest_forecast_tcn, latest_forecast_lstm
    if FORECAST_TCN_PATH.exists():
        with open(FORECAST_TCN_PATH, 'r') as f:
            latest_forecast_tcn = json.load(f)
        print("[INFO] Loaded existing TCN forecast")
    if FORECAST_LSTM_PATH.exists():
        with open(FORECAST_LSTM_PATH, 'r') as f:
            latest_forecast_lstm = json.load(f)
        print("[INFO] Loaded existing LSTM forecast")

    # Start MQTT subscriber
    mqtt_client = start_mqtt_subscriber(args)

    # Start Flask API
    app.config['args'] = args

    print(f"\n[API] Server running at http://{args.api_host}:{args.api_port}")
    print("[API] Endpoints:")
    print("  GET /api/tcn/forecast        — TCN+ARIMA forecast")
    print("  GET /api/tcn/forecast/latest  — TCN h+1")
    print("  GET /api/lstm/forecast        — LSTM+ARIMA forecast")
    print("  GET /api/lstm/forecast/latest — LSTM h+1")
    print("  GET /api/compare             — Compare 2 models")
    print("  GET /api/current             — Latest sensor data")
    print("  GET /api/status              — Health check")
    print("  POST /api/upload_csv         — Upload CSV for testing")
    print("  POST /api/trigger            — Manually trigger inference\n")

    try:
        app.run(host=args.api_host, port=args.api_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        print("[INFO] Server stopped")


if __name__ == '__main__':
    main()