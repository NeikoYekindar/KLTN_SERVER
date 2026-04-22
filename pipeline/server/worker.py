"""
=============================================================================
Inference Worker  —  MQTT + Dual TCN + GCS Upload
=============================================================================
Component 1: receive data from MQTT, run inference, save local + upload GCS.
No Flask API — background worker only.

Usage:
    python server/worker.py \
        --broker   localhost \
        --tcn      models/tcn_model.pth \
        --tcn_hard models/tcn_hard_model.pth \
        --classifier models/condition_classifier.pkl

Env vars:
    GCS_BUCKET                      — GCS bucket name (required for upload)
    GOOGLE_APPLICATION_CREDENTIALS  — path to service account key JSON
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

# GCS paths (relative inside bucket)
GCS_CSV_PATH      = "data/last_48h.csv"
GCS_FORECAST_PATH = "forecasts/forecast_result.json"

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
    print("[DEVICE] No GPU found → cpu")
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

        print(f"\n[DATA] Received {num_rows} rows from {device_id}")

        if not csv_data:
            print("[ERROR] Payload missing csv_data")
            return

        print(f"[DATA] csv_data OK ({len(csv_data)} bytes)")

        # --- Save local ---
        try:
            CSV_PATH.write_text(csv_data, encoding='utf-8')
            print(f"[DATA] Saved local → {CSV_PATH}")
        except Exception as e:
            _log_error(f"Cannot write {CSV_PATH}: {e}")
            return

        try:
            ts           = datetime.now().strftime('%Y%m%d_%H%M%S')
            history_file = HISTORY_DIR / f"data_{ts}_{device_id}.csv"
            history_file.write_text(csv_data, encoding='utf-8')
            print(f"[DATA] Saved history → {history_file}")
        except Exception as e:
            _log_error(f"Cannot write history file: {e}")
            # Continue to inference anyway

        # --- Upload GCS ---
        try:
            gcs_client.upload_file(CSV_PATH, GCS_CSV_PATH)
            gcs_client.upload_string(csv_data, f"data/history/data_{ts}_{device_id}.csv")
        except Exception as gcs_err:
            print(f"[WARN] GCS upload failed (skipping, continuing to inference): {gcs_err}")

        worker_status['last_data_received'] = datetime.now().isoformat()

        # --- Run inference ---
        print("[DATA] Calling run_inference...")
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

    print(f"[INFERENCE] run_inference() called. CSV exists={CSV_PATH.exists()}, lock_locked={inference_lock.locked()}")

    if not CSV_PATH.exists():
        print("[WARN] No CSV file found, skipping inference")
        return

    if not inference_lock.acquire(blocking=False):
        print("[WARN] Inference already running (lock held), skipping new request")
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

        print(f"\n[INFERENCE] Running: {' '.join(cmd)}")
        t0 = time.time()

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        elapsed = time.time() - t0
        print(f"[INFERENCE] Done in {elapsed:.1f}s")

        if result.returncode == 0:
            stdout_tail = result.stdout[-600:] if len(result.stdout) > 600 else result.stdout
            print(stdout_tail)

            if FORECAST_PATH.exists():
                # --- Upload latest forecast to GCS ---
                gcs_client.upload_file(FORECAST_PATH, GCS_FORECAST_PATH)

                # --- Save + upload forecast history ---
                ts_hist   = datetime.now().strftime('%Y%m%d_%H%M%S')
                hist_file = FORECAST_HISTORY_DIR / f"forecast_{ts_hist}.json"

                with open(FORECAST_PATH, 'r', encoding='utf-8') as f:
                    forecast_data = json.load(f)

                history_entry = {
                    'run_id':   ts_hist,
                    'saved_at': datetime.now().isoformat(),
                    **forecast_data,
                }
                with open(hist_file, 'w', encoding='utf-8') as f:
                    json.dump(history_entry, f, ensure_ascii=False, indent=2)

                gcs_client.upload_file(hist_file, f"forecasts/history/forecast_{ts_hist}.json")

                n = len(forecast_data.get('forecast', []))
                print(f"[INFERENCE] {n} steps → GCS uploaded")

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
        print("[WARN] paho-mqtt not installed — MQTT disabled")
        return None

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(args.topic, qos=1)
            print(f"[MQTT] Subscribed → '{args.topic}'")
        else:
            print(f"[MQTT] Connection failed: rc={rc}")

    def on_disconnect(client, userdata, rc):
        if rc != 0:
            print(f"[MQTT] Disconnected (rc={rc}), reconnecting...")

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
        print(f"[MQTT] Connected to {args.broker}:{args.mqtt_port}")
        return client
    except Exception as e:
        print(f"[MQTT] Cannot connect: {e}")
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
            parser.error(f"{label}: file not found")

    if not os.path.exists(args.inference_script):
        parser.error(f"--inference-script: not found '{args.inference_script}'")

    init_dirs()

    bucket = os.environ.get('GCS_BUCKET', '')
    print("=" * 60)
    print("  INFERENCE WORKER  —  Dual TCN + GCS")
    print("=" * 60)
    print(f"  MQTT       : {args.broker}:{args.mqtt_port}")
    print(f"  Topic      : {args.topic}")
    print(f"  TCN        : {args.tcn}")
    print(f"  TCN-Hard   : {args.tcn_hard}")
    print(f"  Classifier : {args.classifier or 'N/A'}")
    print(f"  Device     : {args.device}")
    print(f"  GCS Bucket : {f'gs://{bucket}' if bucket else 'DISABLED (set GCS_BUCKET)'}")
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
        print("\n[INFO] Worker stopped.")


if __name__ == '__main__':
    main()
