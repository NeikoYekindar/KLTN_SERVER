"""
=============================================================================
Inference Worker  —  MQTT + Dual TCN + GCS Upload
=============================================================================
Component 1: receive data from MQTT, run inference, save local + upload GCS.
No Flask API — background worker only.

Usage:
    python server/worker.py \
        --broker   localhost \
        --tcn               models/tcn_model.pth \
        --tcn_hard          models/tcn_hard_model.pth \
        --classifier        models/condition_classifier.pkl \
        --rain_prob_classifier models/rain_prob_classifier.pkl

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
from http.server import BaseHTTPRequestHandler, HTTPServer
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

EFS_BASE = os.environ.get('EFS_BASE', '/mnt/efs/fs1')

MODELS_DIR           = Path("models")
DATA_DIR             = Path(EFS_BASE) / "data"
CSV_PATH             = DATA_DIR / "last_48h.csv"
FORECAST_PATH        = DATA_DIR / "forecast_result.json"
HISTORY_DIR          = DATA_DIR / "history"
FORECAST_HISTORY_DIR = Path(EFS_BASE) / "forecasts" / "history"

# ============================================================
# State
# ============================================================

inference_lock = threading.Lock()
_worker_args   = None   # set in main(), used by /infer HTTP handler

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

        # --- Ghi vào EFS ---
        try:
            CSV_PATH.write_text(csv_data, encoding='utf-8')
            print(f"[DATA] Saved → {CSV_PATH}")
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

        worker_status['last_data_received'] = datetime.now().isoformat()

        # --- Run inference ---
        print("[DATA] Calling run_inference...")
        run_inference(args)   # no custom paths → normal MQTT run

    except json.JSONDecodeError as e:
        _log_error(f"JSON parse error: {e}")
    except Exception as e:
        _log_error(f"process_incoming_data error: {e}")


# ============================================================
# Inference
# ============================================================

def run_inference(args, csv_path=None, output_path=None):
    """
    Run the inference subprocess.
    - csv_path / output_path: override defaults (used for custom simulation).
      When overriding, the lock is acquired with a blocking timeout so the
      caller can wait for the result.  Normal MQTT calls pass no overrides
      and use non-blocking acquire (skip if busy).
    Returns (success: bool, message: str).
    """
    global worker_status

    is_custom        = csv_path is not None
    effective_csv    = Path(csv_path)    if is_custom else CSV_PATH
    effective_output = Path(output_path) if output_path else FORECAST_PATH

    print(f"[INFERENCE] called. csv={effective_csv}, lock_locked={inference_lock.locked()}")

    if not effective_csv.exists():
        print("[WARN] CSV not found, skipping")
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

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        elapsed = time.time() - t0
        print(f"[INFERENCE] Done in {elapsed:.1f}s")

        if result.returncode == 0:
            stdout_tail = result.stdout[-600:] if len(result.stdout) > 600 else result.stdout
            print(stdout_tail)

            if effective_output.exists() and not is_custom:
                ts_hist   = datetime.now().strftime('%Y%m%d_%H%M%S')
                hist_file = FORECAST_HISTORY_DIR / f"forecast_{ts_hist}.json"

                with open(effective_output, 'r', encoding='utf-8') as f:
                    forecast_data = json.load(f)

                history_entry = {
                    'run_id':   ts_hist,
                    'saved_at': datetime.now().isoformat(),
                    **forecast_data,
                }
                hist_file.write_text(json.dumps(history_entry, ensure_ascii=False, indent=2), encoding='utf-8')

                n = len(forecast_data.get('forecast', []))
                print(f"[INFERENCE] {n} steps saved → {hist_file}")

                worker_status['last_inference_run']  = datetime.now().isoformat()
                worker_status['last_inference_time'] = round(elapsed, 2)
                worker_status['inference_count']    += 1

            return True, str(effective_output)

        else:
            msg = f"Inference failed (rc={result.returncode}): {result.stderr[-400:]}"
            _log_error(msg)
            return False, msg

    except subprocess.TimeoutExpired:
        _log_error("Inference timeout (>180s)")
        return False, "Timeout (>180s)"
    except Exception as e:
        _log_error(f"run_inference error: {e}")
        return False, str(e)
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
# Health Check Server
# ============================================================

HEALTH_PORT = int(os.environ.get('HEALTH_PORT', 8080))


def _start_health_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._send(200, {
                'status':          'ok',
                'started_at':      worker_status.get('started_at'),
                'inference_count': worker_status.get('inference_count', 0),
                'last_inference':  worker_status.get('last_inference_run'),
            })

        def do_POST(self):
            if self.path != '/infer':
                self._send(404, {'error': 'Not found'})
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                body   = json.loads(self.rfile.read(length))
                csv_path    = body.get('csv_path')
                output_path = body.get('output_path')

                if not csv_path or not output_path:
                    self._send(400, {'error': 'Missing csv_path or output_path'})
                    return

                if _worker_args is None:
                    self._send(503, {'error': 'Worker not ready'})
                    return

                print(f"[HEALTH] /infer request — csv={csv_path}")
                ok, msg = run_inference(_worker_args, csv_path=csv_path, output_path=output_path)

                if ok and os.path.exists(output_path):
                    with open(output_path, 'r', encoding='utf-8') as f:
                        fd = json.load(f)
                    self._send(200, {'forecast': fd.get('forecast', [])})
                else:
                    self._send(500, {'error': msg})

            except Exception as e:
                self._send(500, {'error': str(e)})

        def _send(self, status, body_dict):
            body = json.dumps(body_dict).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(('0.0.0.0', HEALTH_PORT), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True, name="health")
    t.start()
    print(f"[HEALTH] Listening on port {HEALTH_PORT} (GET /health, POST /infer)")


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
    parser.add_argument('--classifier',            default=None,
                        help='Condition classifier .pkl')
    parser.add_argument('--rain_prob_classifier',  default=None,
                        help='Rain probability classifier .pkl (outputs 0/45/100%%)')
    parser.add_argument('--device',           default=None)

    args = parser.parse_args()

    if args.device is None:
        args.device = auto_detect_device()

    for attr, label in [('tcn', '--tcn'), ('tcn_hard', '--tcn_hard')]:
        if not os.path.exists(getattr(args, attr)):
            parser.error(f"{label}: file not found")

    if args.classifier and not os.path.exists(args.classifier):
        parser.error(f"--classifier: file not found '{args.classifier}'")

    if args.rain_prob_classifier and not os.path.exists(args.rain_prob_classifier):
        parser.error(f"--rain_prob_classifier: file not found '{args.rain_prob_classifier}'")

    if not os.path.exists(args.inference_script):
        parser.error(f"--inference-script: not found '{args.inference_script}'")

    init_dirs()

    print("=" * 60)
    print("  INFERENCE WORKER  —  Dual TCN + GCS")
    print("=" * 60)
    print(f"  MQTT       : {args.broker}:{args.mqtt_port}")
    print(f"  Topic      : {args.topic}")
    print(f"  TCN        : {args.tcn}")
    print(f"  TCN-Hard        : {args.tcn_hard}")
    print(f"  Classifier      : {args.classifier or 'N/A'}")
    print(f"  Rain Prob Class : {args.rain_prob_classifier or 'N/A'}")
    print(f"  Device     : {args.device}")
    print(f"  EFS Base   : {EFS_BASE}")
    print("=" * 60)

    global _worker_args
    _worker_args = args

    worker_status['started_at'] = datetime.now().isoformat()

    _start_health_server()
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