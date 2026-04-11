"""
=============================================================================
Test Pipeline — Chạy toàn bộ pipeline không cần hardware
=============================================================================
Script này mô phỏng:
  1. Tạo file CSV 48h giả lập
  2. POST lên server API (thay thế MQTT)
  3. Kiểm tra kết quả forecast

Cách dùng:
    # Bước 1: Chạy server trước (terminal 1)
    python server/server.py --broker localhost --tcn models/tcn_model.pth

    # Bước 2: Chạy test (terminal 2)
    python test_pipeline.py --server http://localhost:5000

    # Hoặc chỉ tạo CSV test
    python test_pipeline.py --generate-csv-only
=============================================================================
"""

import argparse
import csv
import json
import random
import sys
import time
import numpy as np
from datetime import datetime, timedelta

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False


def generate_test_csv(output_path="test_48h.csv", hours=48):
    """Tạo CSV test 48 giờ."""
    columns = [
        'timestamp', 'temperature', 'feels_like', 'dewpoint',
        'humidity', 'wind_speed', 'gust_speed', 'wind_direction',
        'pressure', 'precipitation', 'rain_probability',
        'uv_index', 'visibility', 'cloud'
    ]

    now = datetime.now()
    rows = []

    for i in range(hours, 0, -1):
        ts = now - timedelta(hours=i)
        hour = ts.hour

        temp = 28 + 5 * np.sin((hour - 6) * np.pi / 12) + random.gauss(0, 0.5)
        hum = max(40, min(100, 75 - 15 * np.sin((hour - 6) * np.pi / 12) + random.gauss(0, 3)))
        pres = 1010 + random.gauss(0, 2)
        ws = max(0, 8 + 5 * random.gauss(0, 1))
        wd = random.randint(0, 359)
        prec = max(0, random.expovariate(2) if random.random() < 0.3 else 0)

        rows.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'temperature': round(temp, 2),
            'feels_like': round(temp + 0.5 * (hum - 50) / 20, 2),
            'dewpoint': round(temp - (100 - hum) / 5, 2),
            'humidity': round(hum, 2),
            'wind_speed': round(ws, 2),
            'gust_speed': round(ws * 1.3, 2),
            'wind_direction': wd,
            'pressure': round(pres, 2),
            'precipitation': round(prec, 4),
            'rain_probability': round(min(100, max(0, prec * 30 + hum * 0.3)), 2),
            'uv_index': round(max(0, 6 * np.sin(max(0, (hour - 6)) * np.pi / 12)) if 6 <= hour <= 18 else 0, 2),
            'visibility': round(max(1, 10 - prec * 2), 2),
            'cloud': round(max(0, min(100, 50 + random.gauss(0, 20))), 2),
        })

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Tạo {output_path}: {len(rows)} hàng")
    return output_path


def test_upload(server_url, csv_path):
    """Upload CSV qua API."""
    if not REQUESTS_OK:
        print("[ERROR] pip install requests")
        return False

    url = f"{server_url}/api/upload_csv"
    with open(csv_path, 'rb') as f:
        resp = requests.post(url, files={'file': f})

    print(f"[UPLOAD] {resp.status_code}: {resp.json()}")
    return resp.status_code == 200


def test_forecast(server_url):
    """Kiểm tra kết quả forecast."""
    if not REQUESTS_OK:
        return

    # Chờ inference xong
    print("[WAIT] Chờ inference...")
    for i in range(30):
        time.sleep(2)
        try:
            resp = requests.get(f"{server_url}/api/forecast", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('forecast'):
                    print(f"\n[OK] Forecast nhận được!")
                    print(f"  Model: {data.get('model_used')}")
                    print(f"  Horizon: {data.get('horizon_hours')}h")
                    print(f"  Steps: {len(data['forecast'])}")
                    print(f"\n  Dự báo h+1:")
                    step1 = data['forecast'][0]
                    for k, v in step1.items():
                        if k not in ('timestamp', 'horizon_step'):
                            print(f"    {k}: {v}")
                    return True
        except Exception:
            pass
        print(f"  Đang chờ... ({(i+1)*2}s)")

    print("[TIMEOUT] Không nhận được forecast sau 60s")
    return False


def test_status(server_url):
    """Kiểm tra server status."""
    if not REQUESTS_OK:
        return

    resp = requests.get(f"{server_url}/api/status")
    print(f"\n[STATUS] {json.dumps(resp.json(), indent=2)}")


def main():
    parser = argparse.ArgumentParser(description="Test Pipeline")
    parser.add_argument('--server', type=str, default='http://localhost:5000')
    parser.add_argument('--generate-csv-only', action='store_true')
    parser.add_argument('--csv-output', type=str, default='test_48h.csv')
    args = parser.parse_args()

    print("=" * 50)
    print("  PIPELINE TEST")
    print("=" * 50)

    # 1. Tạo CSV
    csv_path = generate_test_csv(args.csv_output)

    if args.generate_csv_only:
        print(f"\n[DONE] CSV tạo tại: {csv_path}")
        return

    # 2. Upload
    print(f"\n[TEST] Upload CSV → {args.server}")
    if not test_upload(args.server, csv_path):
        print("[FAIL] Upload thất bại")
        sys.exit(1)

    # 3. Kiểm tra forecast
    if test_forecast(args.server):
        print("\n[PASS] Pipeline hoạt động!")
    else:
        print("\n[FAIL] Không nhận được forecast")
        sys.exit(1)

    # 4. Status
    test_status(args.server)


if __name__ == '__main__':
    main()
