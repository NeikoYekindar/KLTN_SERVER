"""
=============================================================================
Weather Forecast Server
=============================================================================
Chạy trên Cloud/Server:
  1. MQTT Subscriber — nhận CSV từ Edge
  2. Lưu CSV → last_48h.csv + TimescaleDB (optional)
  3. Chạy inference.py → forecast_result.json
  4. Flask API — Unity GET kết quả

Cách dùng:
    python server.py --broker localhost --tcn models/tcn_model.pth \
                     --arima models/arima_model.pkl

    # Chỉ TCN
    python server.py --broker localhost --tcn models/tcn_model.pth

    # Với ensemble weights
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

from flask import Flask, jsonify, request, send_file
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

# Global state
latest_forecast = None
latest_raw_data = None
server_status = {
    'started_at': None,
    'last_data_received': None,
    'last_inference_run': None,
    'inference_count': 0,
    'errors': [],
}

# ============================================================
# 1. Khởi tạo thư mục
# ============================================================

def init_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    HISTORY_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Xử lý dữ liệu nhận từ MQTT
# ============================================================

def process_incoming_data(payload_str, args):
    """
    Xử lý payload JSON từ MQTT:
    - Parse CSV data
    - Lưu file last_48h.csv
    - Lưu bản sao history
    - Chạy inference
    """
    global latest_raw_data, server_status

    try:
        payload = json.loads(payload_str)
        csv_data = payload.get('csv_data', '')
        device_id = payload.get('device_id', 'unknown')
        num_rows = payload.get('num_rows', 0)

        print(f"\n[DATA] Nhận {num_rows} hàng từ {device_id}")

        if not csv_data:
            print("[ERROR] Payload không có csv_data")
            return

        # Lưu CSV chính
        with open(CSV_PATH, 'w') as f:
            f.write(csv_data)
        print(f"[DATA] Đã lưu → {CSV_PATH}")

        # Lưu bản history
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_file = HISTORY_DIR / f"data_{ts}.csv"
        with open(history_file, 'w') as f:
            f.write(csv_data)

        # Parse để lưu vào state
        reader = csv.DictReader(io.StringIO(csv_data))
        rows = list(reader)
        latest_raw_data = {
            'device_id': device_id,
            'received_at': datetime.now().isoformat(),
            'num_rows': len(rows),
            'last_row': rows[-1] if rows else None,
            'columns': list(rows[0].keys()) if rows else [],
        }

        server_status['last_data_received'] = datetime.now().isoformat()

        # Chạy inference
        run_inference(args)

    except json.JSONDecodeError as e:
        err = f"JSON parse error: {e}"
        print(f"[ERROR] {err}")
        server_status['errors'].append({'time': datetime.now().isoformat(), 'error': err})
    except Exception as e:
        err = f"Process error: {e}"
        print(f"[ERROR] {err}")
        server_status['errors'].append({'time': datetime.now().isoformat(), 'error': err})


# ============================================================
# 3. Chạy inference.py
# ============================================================

def run_inference(args):
    """Gọi inference.py subprocess."""
    global latest_forecast, server_status

    if not CSV_PATH.exists():
        print("[WARN] Chưa có file CSV, bỏ qua inference")
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

    print(f"\n[INFERENCE] Chạy: {' '.join(cmd)}")
    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        elapsed = time.time() - start_time
        print(f"[INFERENCE] Xong trong {elapsed:.1f}s")

        if result.returncode == 0:
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

            # Load kết quả
            if FORECAST_PATH.exists():
                with open(FORECAST_PATH, 'r') as f:
                    latest_forecast = json.load(f)
                print(f"[INFERENCE] Loaded forecast: {len(latest_forecast.get('forecast', []))} steps")

            server_status['last_inference_run'] = datetime.now().isoformat()
            server_status['inference_count'] += 1
        else:
            err = f"Inference failed (rc={result.returncode}): {result.stderr[-300:]}"
            print(f"[ERROR] {err}")
            server_status['errors'].append({
                'time': datetime.now().isoformat(), 'error': err
            })

    except subprocess.TimeoutExpired:
        err = "Inference timeout (>120s)"
        print(f"[ERROR] {err}")
        server_status['errors'].append({'time': datetime.now().isoformat(), 'error': err})
    except Exception as e:
        err = f"Inference error: {e}"
        print(f"[ERROR] {err}")
        server_status['errors'].append({'time': datetime.now().isoformat(), 'error': err})


# ============================================================
# 4. MQTT Subscriber
# ============================================================

def start_mqtt_subscriber(args):
    """Khởi động MQTT subscriber thread."""
    if not MQTT_OK:
        print("[WARN] paho-mqtt chưa cài, bỏ qua MQTT subscriber")
        return None

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(args.topic, qos=1)
            print(f"[MQTT] Subscribed → '{args.topic}'")
        else:
            print(f"[MQTT] Kết nối thất bại: rc={rc}")

    def on_message(client, userdata, msg):
        print(f"[MQTT] Nhận message từ topic '{msg.topic}' ({len(msg.payload)} bytes)")
        # Xử lý trong thread riêng để không block MQTT
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
        print(f"[MQTT] Kết nối tới {args.broker}:{args.mqtt_port}")
        return client
    except Exception as e:
        print(f"[MQTT] Không thể kết nối: {e}")
        return None


# ============================================================
# 5. Flask API — Unity đọc kết quả từ đây
# ============================================================

app = Flask(__name__)
CORS(app)  # Cho phép Unity WebGL gọi


@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    """
    Unity gọi endpoint này để lấy kết quả dự báo.

    Response JSON:
    {
        "generated_at": "2026-04-04T15:00:00",
        "horizon_hours": 6,
        "model_used": "TCN+ARIMA ensemble",
        "forecast": [
            {
                "timestamp": "2026-04-04T16:00:00",
                "horizon_step": 1,
                "temperature": 30.5,
                "humidity": 75.2,
                ...
            },
            ...
        ]
    }
    """
    if latest_forecast:
        return jsonify(latest_forecast)

    # Fallback: đọc từ file
    if FORECAST_PATH.exists():
        with open(FORECAST_PATH, 'r') as f:
            return jsonify(json.load(f))

    return jsonify({'error': 'No forecast available yet'}), 404


@app.route('/api/forecast/latest', methods=['GET'])
def get_latest_step():
    """Lấy chỉ bước dự báo gần nhất (h+1)."""
    if latest_forecast and latest_forecast.get('forecast'):
        return jsonify(latest_forecast['forecast'][0])
    return jsonify({'error': 'No forecast available'}), 404


@app.route('/api/forecast/step/<int:step>', methods=['GET'])
def get_forecast_step(step):
    """Lấy dự báo tại step cụ thể (1-6)."""
    if latest_forecast and latest_forecast.get('forecast'):
        for entry in latest_forecast['forecast']:
            if entry.get('horizon_step') == step:
                return jsonify(entry)
    return jsonify({'error': f'Step {step} not found'}), 404


@app.route('/api/current', methods=['GET'])
def get_current_data():
    """Lấy dữ liệu cảm biến mới nhất (hàng cuối CSV)."""
    if latest_raw_data:
        return jsonify(latest_raw_data)
    return jsonify({'error': 'No sensor data yet'}), 404


@app.route('/api/status', methods=['GET'])
def get_status():
    """Health check / status."""
    return jsonify({
        'status': 'running',
        **server_status,
        'has_forecast': latest_forecast is not None,
        'has_sensor_data': latest_raw_data is not None,
    })


@app.route('/api/trigger', methods=['POST'])
def trigger_inference():
    """Trigger inference thủ công (debug)."""
    if CSV_PATH.exists():
        thread = threading.Thread(target=run_inference, args=(app.config['args'],))
        thread.start()
        return jsonify({'message': 'Inference triggered'})
    return jsonify({'error': 'No CSV data available'}), 400


@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """
    Upload CSV trực tiếp (thay thế MQTT khi test).
    Unity hoặc tool khác POST file CSV lên đây.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file.save(str(CSV_PATH))
    print(f"[API] Nhận CSV upload: {file.filename}")

    # Trigger inference
    thread = threading.Thread(target=run_inference, args=(app.config['args'],))
    thread.start()

    return jsonify({'message': 'CSV saved, inference triggered'})


# ============================================================
# 6. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Weather Forecast Server")

    # MQTT
    parser.add_argument('--broker', type=str, default='localhost', help='MQTT broker IP')
    parser.add_argument('--mqtt-port', type=int, default=1883, help='MQTT port')
    parser.add_argument('--topic', type=str, default='weather/data', help='MQTT topic')
    parser.add_argument('--mqtt-username', type=str, default=None)
    parser.add_argument('--mqtt-password', type=str, default=None)

    # Inference
    parser.add_argument('--inference-script', type=str, default='inference.py',
                        help='Path tới inference.py')
    parser.add_argument('--tcn', type=str, default=None, help='TCN model path (.pth)')
    parser.add_argument('--arima', type=str, default=None, help='ARIMA model path (.pkl)')
    parser.add_argument('--ensemble_json', type=str, default=None, help='Ensemble weights JSON')
    parser.add_argument('--w_tcn', type=float, default=None)
    parser.add_argument('--w_arima', type=float, default=None)
    parser.add_argument('--device', type=str, default='cpu', help='TCN device: cpu/cuda')

    # API
    parser.add_argument('--api-port', type=int, default=5000, help='Flask API port')
    parser.add_argument('--api-host', type=str, default='0.0.0.0', help='Flask API host')

    args = parser.parse_args()

    if not args.tcn and not args.arima:
        parser.error("Cần ít nhất --tcn hoặc --arima")

    init_dirs()

    print("=" * 60)
    print("  WEATHER FORECAST SERVER")
    print("=" * 60)
    print(f"  MQTT Broker : {args.broker}:{args.mqtt_port}")
    print(f"  MQTT Topic  : {args.topic}")
    print(f"  TCN Model   : {args.tcn or 'N/A'}")
    print(f"  ARIMA Model : {args.arima or 'N/A'}")
    print(f"  API         : http://{args.api_host}:{args.api_port}")
    print(f"  Inference   : {args.inference_script}")
    print("=" * 60)

    server_status['started_at'] = datetime.now().isoformat()

    # Load forecast cũ nếu có
    global latest_forecast
    if FORECAST_PATH.exists():
        with open(FORECAST_PATH, 'r') as f:
            latest_forecast = json.load(f)
        print(f"[INFO] Loaded existing forecast from {FORECAST_PATH}")

    # Start MQTT subscriber
    mqtt_client = start_mqtt_subscriber(args)

    # Start Flask API
    app.config['args'] = args
    print(f"\n[API] Server đang chạy tại http://{args.api_host}:{args.api_port}")
    print("[API] Unity gọi GET /api/forecast để lấy kết quả\n")

    try:
        app.run(host=args.api_host, port=args.api_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        print("[INFO] Server đã dừng")


if __name__ == '__main__':
    main()
