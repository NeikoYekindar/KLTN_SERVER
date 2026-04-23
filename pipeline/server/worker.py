"""
=============================================================================
Inference Worker  —  MQTT + Dual TCN + GCS Upload
=============================================================================
Component 1: nhận data từ MQTT, chạy inference, lưu local + upload GCS.
Không có Flask API — chỉ là background worker.

Chạy:
    python server/worker.py \
        --broker   localhost \
        --tcn      models/tcn_model.pth \
        --tcn_hard models/tcn_hard_model.pth \
        --classifier models/condition_classifier.pkl

Env vars:
    GCS_BUCKET                      — tên GCS bucket (bắt buộc để upload)
    GOOGLE_APPLICATION_CREDENTIALS  — path tới service account key JSON
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

try:
    import paho.mqtt.client as mqtt
    MQTT_OK = True
except ImportError:
    MQTT_OK = False

import gcs_client

# ============================================================
# Paths
# ============================================================

DATA_DIR             = Path("data")
MODELS_DIR           = Path("models")
CSV_PATH             = DATA_DIR / "last_48h.csv"
FORECAST_PATH        = DATA_DIR / "forecast_result.json"
HISTORY_DIR          = DATA_DIR / "history"
FORECAST_HISTORY_DIR = DATA_DIR / "forecast_history"

# GCS paths (relative trong bucket)
GCS_CSV_PATH      = "data/last_48h.csv"
GCS_FORECAST_PATH = "forecasts/forecast_result.json"

# EFS paths (sao lưu sang AWS EFS)
EFS_BASE         = os.environ.get('EFS_BASE', '/mnt/efs/fs1')
EFS_CSV          = Path(EFS_BASE) / "data" / "last_48h.csv"
EFS_HISTORY_DIR  = Path(EFS_BASE) / "data" / "history"
EFS_FORECAST     = Path(EFS_BASE) / "forecasts" / "forecast_result.json"
EFS_FORECAST_HIST = Path(EFS_BASE) / "forecasts" / "history"

# ============================================================
# State
# ============================================================

inference_lock = threading.Lock()

worker_status = {
    'started_at':          None,
    'last_data_received':  None,
    'last_inference_run':  None,
    'inference_count':     0,
    'last_inference_time': None,
    'errors':              [],
}


def _log_error(msg: str):
    print(f"[ERROR] {msg}")
    worker_status['errors'].append({'time': datetime.now().isoformat(), 'error': msg})
    if len(worker_status['errors']) > 20:
        worker_status['errors'] = worker_status['errors'][-20:]


def _efs_write(efs_path: Path, content: str):
    """Ghi nội dung lên EFS — lỗi chỉ log, không raise."""
    try:
        efs_path.parent.mkdir(parents=True, exist_ok=True)
        efs_path.write_text(content, encoding='utf-8')
        print(f"[EFS] ↑ {efs_path}")
    except Exception as e:
        print(f"[WARN] EFS write failed ({efs_path}): {e}")


# ============================================================
# Init
# ============================================================

def init_dirs():
    for d in [DATA_DIR, MODELS_DIR, HISTORY_DIR, FORECAST_HISTORY_DIR]:
        d.mkdir(exist_ok=True)


def auto_detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[DEVICE] GPU: {name} → cuda")
            return 'cuda'
    except ImportError:
        pass
    print("[DEVICE] Không có GPU → cpu")
    return 'cpu'


# ============================================================
# Process MQTT message
# ============================================================

def process_incoming_data(payload_str: str, args):
    global worker_status

    try:
        payload   = json.loads(payload_str)
        csv_data  = payload.get('csv_data', '')
        device_id = payload.get('device_id', 'unknown')
        num_rows  = payload.get('num_rows', 0)

        print(f"\n[DATA] Nhận {num_rows} hàng từ {device_id}")

        if not csv_data:
            print("[ERROR] Payload không có csv_data")
            return

        # --- Lưu local ---
        CSV_PATH.write_text(csv_data, encoding='utf-8')
        print(f"[DATA] Lưu local → {CSV_PATH}")

        ts           = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_file = HISTORY_DIR / f"data_{ts}_{device_id}.csv"
        history_file.write_text(csv_data, encoding='utf-8')

        # --- Sao lưu EFS ---
        _efs_write(EFS_CSV, csv_data)
        _efs_write(EFS_HISTORY_DIR / f"data_{ts}_{device_id}.csv", csv_data)

        worker_status['last_data_received'] = datetime.now().isoformat()

        # --- Chạy inference ---
        run_inference(args)

    except json.JSONDecodeError as e:
        _log_error(f"JSON parse error: {e}")
    except Exception as e:
        _log_error(f"process_incoming_data error: {e}")


# ============================================================
# Inference
# ============================================================

def run_inference(args):
    global worker_status

    if not CSV_PATH.exists():
        print("[WARN] Chưa có file CSV, bỏ qua inference")
        return

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

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        elapsed = time.time() - t0
        print(f"[INFERENCE] Xong trong {elapsed:.1f}s")

        if result.returncode == 0:
            stdout_tail = result.stdout[-600:] if len(result.stdout) > 600 else result.stdout
            print(stdout_tail)

            if FORECAST_PATH.exists():
                ts_hist   = datetime.now().strftime('%Y%m%d_%H%M%S')
                hist_file = FORECAST_HISTORY_DIR / f"forecast_{ts_hist}.json"

                with open(FORECAST_PATH, 'r', encoding='utf-8') as f:
                    forecast_data = json.load(f)

                history_entry = {
                    'run_id':   ts_hist,
                    'saved_at': datetime.now().isoformat(),
                    **forecast_data,
                }
                hist_content = json.dumps(history_entry, ensure_ascii=False, indent=2)

                # --- Lưu local ---
                hist_file.write_text(hist_content, encoding='utf-8')

                # --- Sao lưu EFS ---
                _efs_write(EFS_FORECAST, FORECAST_PATH.read_text(encoding='utf-8'))
                _efs_write(EFS_FORECAST_HIST / f"forecast_{ts_hist}.json", hist_content)

                n = len(forecast_data.get('forecast', []))
                print(f"[INFERENCE] {n} steps saved local + EFS")

            worker_status['last_inference_run']  = datetime.now().isoformat()
            worker_status['last_inference_time'] = round(elapsed, 2)
            worker_status['inference_count']    += 1

        else:
            _log_error(f"Inference failed (rc={result.returncode}): {result.stderr[-400:]}")

    except subprocess.TimeoutExpired:
        _log_error("Inference timeout (>180s)")
    except Exception as e:
        _log_error(f"run_inference error: {e}")
    finally:
        inference_lock.release()


# ============================================================
# MQTT
# ============================================================

def start_mqtt_subscriber(args):
    if not MQTT_OK:
        print("[WARN] paho-mqtt chưa cài — không có MQTT")
        return None

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(args.topic, qos=1)
            print(f"[MQTT] Subscribed → '{args.topic}'")
        else:
            print(f"[MQTT] Kết nối thất bại: rc={rc}")

    def on_disconnect(client, userdata, rc):
        if rc != 0:
            print(f"[MQTT] Mất kết nối (rc={rc}), đang reconnect...")

    def on_message(client, userdata, msg):
        print(f"[MQTT] Message topic='{msg.topic}' size={len(msg.payload)}B")
        threading.Thread(
            target=process_incoming_data,
            args=(msg.payload.decode('utf-8'), args),
            daemon=True,
        ).start()

    client = mqtt.Client(client_id=f"worker_{os.getpid()}", protocol=mqtt.MQTTv311)
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
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Inference Worker — Dual TCN + GCS")

    parser.add_argument('--broker',           default='localhost')
    parser.add_argument('--mqtt-port',        type=int, default=1883)
    parser.add_argument('--topic',            default='weather/data')
    parser.add_argument('--mqtt-username',    default=None)
    parser.add_argument('--mqtt-password',    default=None)

    parser.add_argument('--inference-script', default='server/inference_dual_tcn.py')
    parser.add_argument('--tcn',              required=True)
    parser.add_argument('--tcn_hard',         required=True)
    parser.add_argument('--classifier',       default=None)
    parser.add_argument('--device',           default=None)

    args = parser.parse_args()

    if args.device is None:
        args.device = auto_detect_device()

    for attr, label in [('tcn', '--tcn'), ('tcn_hard', '--tcn_hard')]:
        if not os.path.exists(getattr(args, attr)):
            parser.error(f"{label}: file không tồn tại")

    if not os.path.exists(args.inference_script):
        parser.error(f"--inference-script: không tìm thấy '{args.inference_script}'")

    init_dirs()

    print("=" * 60)
    print("  INFERENCE WORKER  —  Dual TCN + GCS")
    print("=" * 60)
    print(f"  MQTT       : {args.broker}:{args.mqtt_port}")
    print(f"  Topic      : {args.topic}")
    print(f"  TCN        : {args.tcn}")
    print(f"  TCN-Hard   : {args.tcn_hard}")
    print(f"  Classifier : {args.classifier or 'N/A'}")
    print(f"  Device     : {args.device}")
    print(f"  EFS Base   : {EFS_BASE}")
    print("=" * 60)

    worker_status['started_at'] = datetime.now().isoformat()

    mqtt_client = start_mqtt_subscriber(args)

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        print("\n[INFO] Worker dừng.")


if __name__ == '__main__':
    main()
