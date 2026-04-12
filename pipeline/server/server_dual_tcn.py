"""
=============================================================================
Weather Forecast Server  —  Dual TCN Edition
=============================================================================
Chạy trên Cloud/Server:
  1. MQTT Subscriber  — nhận CSV từ Edge
  2. Lưu CSV → last_48h.csv + history/
  3. Chạy inference_dual_tcn.py → forecast_result.json
  4. Flask API          — Unity GET kết quả

Cách dùng:
    python server.py \
        --broker   localhost \
        --tcn      models/tcn_model_24h.pth \
        --tcn_hard models/tcn_hard_model_24h.pth

    # Chỉ định topic + device
    python server.py \
        --broker   192.168.1.100 \
        --tcn      models/tcn_model_24h.pth \
        --tcn_hard models/tcn_hard_model_24h.pth \
        --topic    weather/sgn \
        --device   cuda

    # Custom port API
    python server.py \
        --broker   localhost \
        --tcn      models/tcn_model_24h.pth \
        --tcn_hard models/tcn_hard_model_24h.pth \
        --api-port 8080

Endpoints:
    GET  /api/forecast           — toàn bộ 24h dự báo
    GET  /api/forecast/latest    — chỉ h+1
    GET  /api/forecast/step/<n>  — step n (1–24)
    GET  /api/current            — dữ liệu cảm biến mới nhất
    GET  /api/status             — health check
    POST /api/trigger            — chạy inference thủ công
    POST /api/upload_csv         — upload CSV (thay thế MQTT khi test)
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
# Device auto-detect
# ============================================================

def auto_detect_device() -> str:
    """Tự động chọn cuda nếu có GPU, ngược lại dùng cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[DEVICE] GPU detected: {name} → dùng cuda")
            return 'cuda'
    except ImportError:
        pass
    print("[DEVICE] Không tìm thấy GPU → dùng cpu")
    return 'cpu'


# ============================================================
# Paths
# ============================================================

DATA_DIR     = Path("data")
MODELS_DIR   = Path("models")
CSV_PATH     = DATA_DIR / "last_48h.csv"
FORECAST_PATH = DATA_DIR / "forecast_result.json"
HISTORY_DIR  = DATA_DIR / "history"
FORECAST_HISTORY_DIR = DATA_DIR / "forecast_history"
# ============================================================
# Global state
# ============================================================

latest_forecast  = None
latest_raw_data  = None
inference_lock   = threading.Lock()   # tránh chạy inference đồng thời

server_status = {
    'started_at':          None,
    'last_data_received':  None,
    'last_inference_run':  None,
    'inference_count':     0,
    'last_inference_time': None,   # giây
    'errors':              [],     # giữ tối đa 20 lỗi gần nhất
}


# ============================================================
# 1. Khởi tạo thư mục
# ============================================================

def init_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    HISTORY_DIR.mkdir(exist_ok=True)
    FORECAST_HISTORY_DIR.mkdir(exist_ok=True)

# ============================================================
# 2. Xử lý dữ liệu nhận từ MQTT
# ============================================================

def process_incoming_data(payload_str: str, args):
    """
    Parse payload JSON từ MQTT:
      { "csv_data": "...", "device_id": "edge_01", "num_rows": 48 }
    Lưu CSV → chạy inference.
    """
    global latest_raw_data, server_status

    try:
        payload   = json.loads(payload_str)
        csv_data  = payload.get('csv_data', '')
        device_id = payload.get('device_id', 'unknown')
        num_rows  = payload.get('num_rows', 0)

        print(f"\n[DATA] Nhận {num_rows} hàng từ {device_id}")

        if not csv_data:
            print("[ERROR] Payload không có csv_data")
            return

        # Lưu CSV chính
        CSV_PATH.write_text(csv_data, encoding='utf-8')
        print(f"[DATA] Đã lưu → {CSV_PATH}")

        # Lưu bản history
        ts           = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_file = HISTORY_DIR / f"data_{ts}_{device_id}.csv"
        history_file.write_text(csv_data, encoding='utf-8')

        # Parse để lưu raw state
        reader = csv.DictReader(io.StringIO(csv_data))
        rows   = list(reader)
        latest_raw_data = {
            'device_id':   device_id,
            'received_at': datetime.now().isoformat(),
            'num_rows':    len(rows),
            'last_row':    rows[-1] if rows else None,
            'columns':     list(rows[0].keys()) if rows else [],
        }
        server_status['last_data_received'] = datetime.now().isoformat()

        # Chạy inference
        run_inference(args)

    except json.JSONDecodeError as e:
        _log_error(f"JSON parse error: {e}")
    except Exception as e:
        _log_error(f"process_incoming_data error: {e}")


# ============================================================
# 3. Chạy inference_dual_tcn.py
# ============================================================

def run_inference(args):
    """
    Gọi inference_dual_tcn.py qua subprocess.
    Dùng lock để tránh chạy song song.
    """
    global latest_forecast, server_status

    if not CSV_PATH.exists():
        print("[WARN] Chưa có file CSV, bỏ qua inference")
        return

    # Nếu đang inference thì bỏ qua lần này
    if not inference_lock.acquire(blocking=False):
        print("[WARN] Inference đang chạy, bỏ qua request mới")
        return

    try:
        cmd = [
            sys.executable, args.inference_script,
            '--tcn',      args.tcn,
            '--tcn_hard', args.tcn_hard,
            '--csv',      str(CSV_PATH),
            '--output',   str(FORECAST_PATH),
            '--device',   args.device,
        ]
        if args.classifier:
            cmd.extend(['--classifier', args.classifier])

        print(f"\n[INFERENCE] Chạy: {' '.join(cmd)}")
        t0 = time.time()

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180
        )

        elapsed = time.time() - t0
        print(f"[INFERENCE] Xong trong {elapsed:.1f}s")

        if result.returncode == 0:
            # In phần cuối stdout (tóm tắt)
            stdout_tail = result.stdout[-800:] if len(result.stdout) > 800 else result.stdout
            print(stdout_tail)

            # Load forecast mới
            if FORECAST_PATH.exists():
                with open(FORECAST_PATH, 'r', encoding='utf-8') as f:
                    latest_forecast = json.load(f)
                n = len(latest_forecast.get('forecast', []))
                print(f"[INFERENCE] Loaded forecast: {n} steps")
                # Lưu bản forecast history
                ts_hist  = datetime.now().strftime('%Y%m%d_%H%M%S')
                hist_file = FORECAST_HISTORY_DIR / f"forecast_{ts_hist}.json"

                # Thêm trường run_id và saved_at vào bản history
                history_entry = {
                    'run_id':   ts_hist,
                    'saved_at': datetime.now().isoformat(),
                    **latest_forecast,
                }
                with open(hist_file, 'w', encoding='utf-8') as f:
                    json.dump(history_entry, f, ensure_ascii=False, indent=2)
                print(f"[INFERENCE] Forecast history saved → {hist_file}")

            server_status['last_inference_run']  = datetime.now().isoformat()
            server_status['last_inference_time'] = round(elapsed, 2)
            server_status['inference_count']    += 1

        else:
            _log_error(
                f"Inference failed (rc={result.returncode}): "
                f"{result.stderr[-400:]}"
            )

    except subprocess.TimeoutExpired:
        _log_error("Inference timeout (>180s)")
    except Exception as e:
        _log_error(f"run_inference error: {e}")
    finally:
        inference_lock.release()


def _log_error(msg: str):
    print(f"[ERROR] {msg}")
    server_status['errors'].append({
        'time':  datetime.now().isoformat(),
        'error': msg,
    })
    # Giữ tối đa 20 lỗi gần nhất
    if len(server_status['errors']) > 20:
        server_status['errors'] = server_status['errors'][-20:]


# ============================================================
# 4. MQTT Subscriber
# ============================================================

def start_mqtt_subscriber(args):
    if not MQTT_OK:
        print("[WARN] paho-mqtt chưa được cài — bỏ qua MQTT subscriber")
        print("       pip install paho-mqtt")
        return None

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(args.topic, qos=1)
            print(f"[MQTT] Subscribed → '{args.topic}'")
        else:
            print(f"[MQTT] Kết nối thất bại: rc={rc}")

    def on_disconnect(client, userdata, rc):
        if rc != 0:
            print(f"[MQTT] Mất kết nối (rc={rc}), đang tự reconnect...")

    def on_message(client, userdata, msg):
        print(f"[MQTT] Nhận message  topic='{msg.topic}'  size={len(msg.payload)}B")
        threading.Thread(
            target=process_incoming_data,
            args=(msg.payload.decode('utf-8'), args),
            daemon=True,
        ).start()

    client = mqtt.Client(
        client_id=f"forecast_server_{os.getpid()}",
        protocol=mqtt.MQTTv311,
    )

    if args.mqtt_username:
        client.username_pw_set(args.mqtt_username, args.mqtt_password)

    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_message    = on_message
    client.reconnect_delay_set(min_delay=2, max_delay=60)

    try:
        client.connect(args.broker, args.mqtt_port, keepalive=120)
        client.loop_start()
        print(f"[MQTT] Kết nối tới {args.broker}:{args.mqtt_port}")
        return client
    except Exception as e:
        print(f"[MQTT] Không thể kết nối: {e}")
        return None


# ============================================================
# 5. Flask API
# ============================================================

app = Flask(__name__)
CORS(app)   # Unity WebGL cần header CORS


# ----------------------------------------------------------
# GET /api/forecast
# Trả về toàn bộ 24h dự báo
# ----------------------------------------------------------
@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    """
    Trả về toàn bộ dự báo 24h.

    Response:
    {
        "generated_at": "...",
        "horizon_hours": 24,
        "model_used": "TCN + TCN-Hard (dual model)",
        "targets": { "tcn": [...], "tcn_hard": [...] },
        "target_cols": [...],
        "forecast": [
            {
                "timestamp": "2026-04-06T10:00:00",
                "horizon_step": 1,
                "temperature": 31.2,
                "humidity": 74.5,
                "wind_speed": 12.3,
                "pressure": 1010.2,
                "uv_index": 5.1,
                "dewpoint": 23.4,
                "visibility": 9.8,
                "wind_direction": 135.0,
                "precip_has_rain": 0.72,
                "precip_amount": 0.45,
                "rain_probability": 68.3,
                "gust_speed": 18.7,
                "cloud": 62.1
            },
            ...  (24 entries)
        ]
    }
    """
    data = _load_forecast()
    if data:
        return jsonify(data)
    return jsonify({'error': 'No forecast available yet'}), 404


# ----------------------------------------------------------
# GET /api/forecast/latest
# Chỉ trả về step h+1
# ----------------------------------------------------------
@app.route('/api/forecast/latest', methods=['GET'])
def get_latest_step():
    """Trả về chỉ bước dự báo h+1 (1 giờ tới)."""
    data = _load_forecast()
    if data and data.get('forecast'):
        return jsonify(data['forecast'][0])
    return jsonify({'error': 'No forecast available'}), 404


# ----------------------------------------------------------
# GET /api/forecast/step/<n>
# Trả về step cụ thể (1–24)
# ----------------------------------------------------------
@app.route('/api/forecast/step/<int:step>', methods=['GET'])
def get_forecast_step(step):
    """Trả về dự báo tại step n (1–24)."""
    data = _load_forecast()
    if data and data.get('forecast'):
        for entry in data['forecast']:
            if entry.get('horizon_step') == step:
                return jsonify(entry)
        return jsonify({'error': f'Step {step} không tìm thấy (1–24)'}), 404
    return jsonify({'error': 'No forecast available'}), 404


# ----------------------------------------------------------
# GET /api/current
# Dữ liệu cảm biến mới nhất (hàng cuối CSV)
# ----------------------------------------------------------
@app.route('/api/current', methods=['GET'])
def get_current_data():
    """Trả về metadata của dữ liệu cảm biến mới nhất."""
    if latest_raw_data:
        return jsonify(latest_raw_data)
    return jsonify({'error': 'Chưa có dữ liệu cảm biến'}), 404


# ----------------------------------------------------------
# GET /api/status
# Health check
# ----------------------------------------------------------
@app.route('/api/status', methods=['GET'])
def get_status():
    """Trạng thái server — dùng để monitor."""
    return jsonify({
        'status':          'running',
        'has_forecast':    latest_forecast is not None or FORECAST_PATH.exists(),
        'has_sensor_data': latest_raw_data is not None,
        **server_status,
    })


# ----------------------------------------------------------
# POST /api/trigger
# Trigger inference thủ công
# ----------------------------------------------------------
@app.route('/api/trigger', methods=['POST'])
def trigger_inference():
    """
    Trigger inference thủ công — dùng để debug hoặc cron.
    Cần có file CSV đã tồn tại.
    """
    if not CSV_PATH.exists():
        return jsonify({'error': 'Chưa có file CSV, hãy upload trước'}), 400

    threading.Thread(
        target=run_inference,
        args=(app.config['args'],),
        daemon=True,
    ).start()

    return jsonify({'message': 'Inference triggered', 'csv': str(CSV_PATH)})


# ----------------------------------------------------------
# POST /api/upload_csv
# Upload CSV trực tiếp (thay MQTT khi test)
# ----------------------------------------------------------
@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """
    Upload file CSV trực tiếp thay vì qua MQTT.
    Hữu ích khi test local hoặc dùng Postman.

    Form-data: file=<csv_file>
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Thiếu field "file" trong form-data'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Chỉ chấp nhận file .csv'}), 400

    file.save(str(CSV_PATH))
    print(f"[API] Nhận CSV upload: {file.filename} → {CSV_PATH}")

    threading.Thread(
        target=run_inference,
        args=(app.config['args'],),
        daemon=True,
    ).start()

    return jsonify({
        'message':  'CSV đã lưu, inference đang chạy',
        'filename': file.filename,
        'saved_to': str(CSV_PATH),
    })


# ----------------------------------------------------------
# Helper: load forecast (memory → file fallback)
# ----------------------------------------------------------
def _load_forecast():
    global latest_forecast
    if latest_forecast:
        return latest_forecast
    if FORECAST_PATH.exists():
        with open(FORECAST_PATH, 'r', encoding='utf-8') as f:
            latest_forecast = json.load(f)
        return latest_forecast
    return None

# ----------------------------------------------------------
# GET /api/forecast/history
# Danh sách các lần dự báo (metadata, không kèm data nặng)
# ----------------------------------------------------------
@app.route('/api/forecast/history', methods=['GET'])
def get_forecast_history():
    """
    Trả về danh sách các lần dự báo đã lưu.
    Query params:
        ?limit=20   — số lượng tối đa (mặc định 20)
    """
    limit = request.args.get('limit', 20, type=int)
    files = sorted(FORECAST_HISTORY_DIR.glob('forecast_*.json'), reverse=True)

    history = []
    for fp in files[:limit]:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            history.append({
                'run_id':        data.get('run_id', fp.stem),
                'saved_at':      data.get('saved_at'),
                'generated_at':  data.get('generated_at'),
                'horizon_hours': data.get('horizon_hours'),
                'num_steps':     len(data.get('forecast', [])),
                #'temperature':   data.get('forecast', [{}])[1].get('temperature'),
                'filename':      fp.name,
            })
        except Exception:
            continue
 
    return jsonify({'count': len(history), 'history': history})

# ----------------------------------------------------------
# GET /api/forecast/history/<run_id>
# Chi tiết một lần dự báo cụ thể
# ----------------------------------------------------------
@app.route('/api/forecast/history/<run_id>', methods=['GET'])
def get_forecast_history_detail(run_id):
    """Trả về toàn bộ dữ liệu của một lần dự báo theo run_id."""
    fp = FORECAST_HISTORY_DIR / f"forecast_{run_id}.json"
    if not fp.exists():
        return jsonify({'error': f'Forecast run_id "{run_id}" not found'}), 404
    with open(fp, 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))



# ============================================================
# 6. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Weather Forecast Server — Dual TCN (tcn + tcn_hard)"
    )

    # MQTT
    parser.add_argument('--broker',        type=str,   default='localhost',
                        help='MQTT broker IP/hostname (mặc định: localhost)')
    parser.add_argument('--mqtt-port',     type=int,   default=1883,
                        help='MQTT port (mặc định: 1883)')
    parser.add_argument('--topic',         type=str,   default='weather/data',
                        help='MQTT topic lắng nghe (mặc định: weather/data)')
    parser.add_argument('--mqtt-username', type=str,   default=None)
    parser.add_argument('--mqtt-password', type=str,   default=None)

    # Inference
    parser.add_argument('--inference-script', type=str, default='server/inference_dual_tcn.py',
                        help='Path tới inference_dual_tcn.py')
    parser.add_argument('--tcn',      type=str, required=True,
                        help='TCN model path (.pth) — temperature/humidity/...')
    parser.add_argument('--tcn_hard', type=str, required=True,
                        help='TCN-Hard model path (.pth) — wind_direction/precip/...')
    parser.add_argument('--classifier', type=str, default=None,
                        help='Condition classifier path (.pkl) — predict weather condition')
    parser.add_argument('--device',   type=str, default=None,
                        help='PyTorch device: cpu hoặc cuda (mặc định: tự detect)')

    # API
    parser.add_argument('--api-port', type=int, default=5000,
                        help='Flask API port (mặc định: 5000)')
    parser.add_argument('--api-host', type=str, default='0.0.0.0',
                        help='Flask API host (mặc định: 0.0.0.0)')

    args = parser.parse_args()

    # Auto-detect device nếu không chỉ định
    if args.device is None:
        args.device = auto_detect_device()
    else:
        print(f"[DEVICE] Dùng device chỉ định: {args.device}")

    # Validate model files tồn tại
    for attr, label in [('tcn', '--tcn'), ('tcn_hard', '--tcn_hard')]:
        path = getattr(args, attr)
        if not os.path.exists(path):
            parser.error(f"{label}: file không tồn tại → '{path}'")

    # Validate inference script
    if not os.path.exists(args.inference_script):
        parser.error(
            f"--inference-script: không tìm thấy '{args.inference_script}'\n"
            f"Hãy đặt inference_dual_tcn.py cùng thư mục hoặc chỉ định đường dẫn đúng."
        )

    init_dirs()

    print("=" * 65)
    print("  WEATHER FORECAST SERVER  —  Dual TCN Edition")
    print("=" * 65)
    print(f"  MQTT Broker      : {args.broker}:{args.mqtt_port}")
    print(f"  MQTT Topic       : {args.topic}")
    print(f"  TCN model        : {args.tcn}")
    print(f"  TCN-Hard model   : {args.tcn_hard}")
    print(f"  Inference script : {args.inference_script}")
    print(f"  Device           : {args.device} {'(auto)' if '--device' not in sys.argv else '(manual)'}")
    print(f"  API              : http://{args.api_host}:{args.api_port}")
    print("=" * 65)

    server_status['started_at'] = datetime.now().isoformat()

    # Load forecast cũ nếu có sẵn
    global latest_forecast
    if FORECAST_PATH.exists():
        with open(FORECAST_PATH, 'r', encoding='utf-8') as f:
            latest_forecast = json.load(f)
        print(f"[INFO] Loaded existing forecast từ {FORECAST_PATH}")

    # Start MQTT subscriber
    mqtt_client = start_mqtt_subscriber(args)

    # Truyền args vào Flask config để dùng trong endpoint /api/trigger
    app.config['args'] = args

    print(f"\n[API] Server running at http://{args.api_host}:{args.api_port}")
    print("[API] Endpoints:")
    print("       GET  /api/forecast           — 24h")
    print("       GET  /api/forecast/latest    — h+1 (1 h next hour)")
    print("       GET  /api/forecast/step/<n>  — step n (1–24)")
    print("       GET  /api/forecast/history   — list all past forecasts")
    print("       GET  /api/forecast/history/<run_id> — detail of a past forecast")
    print("       GET  /api/current            — data sensor latest (metadata)")
    print("       GET  /api/status             — health check")
    print("       POST /api/trigger            — run inference manually (need existing CSV)")
    print("       POST /api/upload_csv         — upload CSV (test don't need MQTT)\n")

    try:
        app.run(host=args.api_host, port=args.api_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        print("\n[INFO] Server đã dừng.")


if __name__ == '__main__':
    main()
