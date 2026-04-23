"""
=============================================================================
Edge Publisher — Chạy trên Raspberry Pi 5
=============================================================================
- Đọc dữ liệu từ cảm biến (hoặc CSV mô phỏng)
- Duy trì buffer 48 giờ gần nhất
- Publish CSV qua MQTT mỗi chu kỳ (mặc định 1 giờ)

Cách dùng:
    # Chạy với dữ liệu mô phỏng
    python edge_publisher.py --broker 192.168.1.100 --simulate

    # Chạy với cảm biến thật
    python edge_publisher.py --broker 192.168.1.100

    # Tùy chỉnh
    python edge_publisher.py --broker your-server-ip --port 1883 \
                             --topic weather/data --interval 3600
=============================================================================
"""

import argparse
import csv
import io
import json
import os
import time
import random
import numpy as np
from datetime import datetime, timedelta

try:
    import paho.mqtt.client as mqtt
    MQTT_OK = True
except ImportError:
    MQTT_OK = False
    print("[WARN] paho-mqtt chưa cài. pip install paho-mqtt")


# ============================================================
# 1. Cảm biến thật (uncomment khi chạy trên Pi)
# ============================================================

# import board
# import adafruit_dht
# import RPi.GPIO as GPIO

def read_real_sensors():
    """
    Đọc cảm biến thật trên Raspberry Pi.
    Tùy chỉnh theo hardware của bạn.
    """
    # DHT22 - nhiệt độ & độ ẩm
    # dht = adafruit_dht.DHT22(board.D4)
    # temperature = dht.temperature
    # humidity = dht.humidity

    # Ultrasonic HC-SR04 - mực nước
    # ... đọc GPIO

    # Anemometer - tốc độ gió
    # ... đọc xung

    # Tipping bucket rain gauge - lượng mưa
    # ... đếm xung

    # Placeholder - thay bằng code cảm biến thật
    raise NotImplementedError(
        "Hãy implement hàm này với cảm biến thật của bạn. "
        "Trả về dict với các key: timestamp, temperature, humidity, "
        "pressure, wind_speed, wind_direction, precipitation, ..."
    )


# ============================================================
# 2. Mô phỏng cảm biến (để test pipeline)
# ============================================================

def simulate_sensor_reading(base_time=None):
    """Tạo dữ liệu giả lập để test pipeline."""
    ts = base_time or datetime.now()
    hour = ts.hour

    # Mô phỏng nhiệt độ theo chu kỳ ngày
    temp_base = 28 + 5 * np.sin((hour - 6) * np.pi / 12)
    temperature = temp_base + random.gauss(0, 0.5)

    humidity = max(40, min(100, 75 - 15 * np.sin((hour - 6) * np.pi / 12) + random.gauss(0, 3)))
    pressure = 1010 + random.gauss(0, 2)
    wind_speed = max(0, 8 + 5 * random.gauss(0, 1))
    wind_direction = random.randint(0, 359)
    precipitation = max(0, random.expovariate(2) if random.random() < 0.3 else 0)
    feels_like = temperature + 0.5 * (humidity - 50) / 20
    dewpoint = temperature - (100 - humidity) / 5
    cloud = max(0, min(100, 50 + random.gauss(0, 20)))
    visibility = max(1, 10 - precipitation * 2 + random.gauss(0, 1))
    uv_index = max(0, 6 * np.sin(max(0, (hour - 6)) * np.pi / 12)) if 6 <= hour <= 18 else 0
    rain_probability = min(100, max(0, precipitation * 30 + humidity * 0.3 + random.gauss(0, 10)))
    gust_speed = wind_speed * (1 + 0.3 * abs(random.gauss(0, 1)))

    return {
        'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
        'temperature': round(temperature, 2),
        'feels_like': round(feels_like, 2),
        'dewpoint': round(dewpoint, 2),
        'humidity': round(humidity, 2),
        'wind_speed': round(wind_speed, 2),
        'gust_speed': round(gust_speed, 2),
        'wind_direction': wind_direction,
        'pressure': round(pressure, 2),
        'precipitation': round(precipitation, 4),
        'rain_probability': round(rain_probability, 2),
        'uv_index': round(uv_index, 2),
        'visibility': round(visibility, 2),
        'cloud': round(cloud, 2),
    }


# ============================================================
# 3. Buffer 48h
# ============================================================

BUFFER_FILE = "sensor_buffer_48h.csv"
BUFFER_SIZE = 48  # 48 hàng = 48 giờ

COLUMNS = [
    'timestamp', 'temperature', 'feels_like', 'dewpoint',
    'humidity', 'wind_speed', 'gust_speed', 'wind_direction',
    'pressure', 'precipitation', 'rain_probability',
    'uv_index', 'visibility', 'cloud'
]


def load_buffer():
    """Load buffer từ file CSV."""
    rows = []
    if os.path.exists(BUFFER_FILE):
        with open(BUFFER_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows[-BUFFER_SIZE:]  # Giữ 48 hàng cuối


def save_buffer(rows):
    """Lưu buffer ra file CSV."""
    with open(BUFFER_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows[-BUFFER_SIZE:])


def append_reading(rows, reading):
    """Thêm một reading vào buffer."""
    rows.append(reading)
    if len(rows) > BUFFER_SIZE:
        rows = rows[-BUFFER_SIZE:]
    return rows


def buffer_to_csv_string(rows):
    """Chuyển buffer thành CSV string để gửi qua MQTT."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


# ============================================================
# 4. MQTT Publisher
# ============================================================

def create_mqtt_client(broker, port, username=None, password=None):
    """Tạo MQTT client."""
    if not MQTT_OK:
        raise ImportError("paho-mqtt chưa được cài")

    client = mqtt.Client(client_id=f"edge_pi_{os.getpid()}", protocol=mqtt.MQTTv311)

    if username:
        client.username_pw_set(username, password)

    # Callbacks
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"[MQTT] Kết nối thành công tới {broker}:{port}")
        else:
            print(f"[MQTT] Lỗi kết nối: rc={rc}")

    def on_disconnect(client, userdata, rc):
        print(f"[MQTT] Mất kết nối (rc={rc}). Sẽ tự kết nối lại...")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    # Auto reconnect
    client.reconnect_delay_set(min_delay=1, max_delay=60)

    client.connect(broker, port, keepalive=120)
    client.loop_start()
    return client


def publish_data(client, topic, csv_data, metadata=None):
    """
    Publish dữ liệu CSV qua MQTT.
    Gửi dưới dạng JSON với CSV string bên trong.
    """
    payload = {
        'type': 'sensor_data',
        'device_id': 'pi_station_01',
        'timestamp': datetime.now().isoformat(),
        'num_rows': csv_data.count('\n') - 1,  # trừ header
        'csv_data': csv_data,
    }
    if metadata:
        payload['metadata'] = metadata

    msg = json.dumps(payload)
    result = client.publish(topic, msg, qos=1)

    if result.rc == 0:
        print(f"[MQTT] Đã publish {payload['num_rows']} hàng → topic '{topic}'")
    else:
        print(f"[MQTT] Publish thất bại: rc={result.rc}")

    return result


# ============================================================
# 5. Main loop
# ============================================================

def generate_initial_buffer(hours=48):
    """Tạo buffer 48h ban đầu (cho simulate mode)."""
    rows = []
    now = datetime.now()
    for i in range(hours, 0, -1):
        ts = now - timedelta(hours=i)
        reading = simulate_sensor_reading(ts)
        rows.append(reading)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Edge Publisher — Raspberry Pi → MQTT")
    parser.add_argument('--broker', type=str, required=True, help='MQTT broker IP')
    parser.add_argument('--port', type=int, default=1883, help='MQTT port (default: 1883)')
    parser.add_argument('--topic', type=str, default='weather/data', help='MQTT topic')
    parser.add_argument('--username', type=str, default=None, help='MQTT username')
    parser.add_argument('--password', type=str, default=None, help='MQTT password')
    parser.add_argument('--interval', type=int, default=3600,
                        help='Chu kỳ gửi dữ liệu (giây, default: 3600 = 1h)')
    parser.add_argument('--simulate', action='store_true',
                        help='Dùng dữ liệu mô phỏng thay vì cảm biến thật')
    parser.add_argument('--init-buffer', action='store_true',
                        help='Tạo buffer 48h ban đầu (simulate mode)')
    parser.add_argument('--send-file', type=str, default=None,
                        help='Gửi thủ công một file CSV lên server rồi thoát')
    args = parser.parse_args()

    print("=" * 60)
    print("  EDGE PUBLISHER — Raspberry Pi 5")
    print(f"  Broker: {args.broker}:{args.port}")
    print(f"  Topic:  {args.topic}")
    print(f"  Mode:   {'Send file' if args.send_file else ('Simulate' if args.simulate else 'Real sensors')}")
    print(f"  Interval: {args.interval}s")
    print("=" * 60)

    # Chế độ gửi thủ công một file CSV rồi thoát
    if args.send_file:
        if not os.path.exists(args.send_file):
            print(f"[ERROR] File không tồn tại: {args.send_file}")
            return
        with open(args.send_file, 'r') as f:
            csv_data = f.read()
        num_rows = csv_data.count('\n') - 1
        print(f"[INFO] Đọc file: {args.send_file} ({num_rows} hàng)")
        client = create_mqtt_client(args.broker, args.port, args.username, args.password)
        time.sleep(2)
        publish_data(client, args.topic, csv_data, metadata={'source_file': os.path.basename(args.send_file)})
        client.loop_stop()
        client.disconnect()
        print("[INFO] Đã gửi xong. Thoát.")
        return

    # Load hoặc tạo buffer
    if args.init_buffer and args.simulate:
        print("[INFO] Tạo buffer 48h ban đầu...")
        buffer = generate_initial_buffer(48)
        save_buffer(buffer)
        print(f"[INFO] Buffer: {len(buffer)} hàng")
    else:
        buffer = load_buffer()
        print(f"[INFO] Load buffer: {len(buffer)} hàng")

    # Kết nối MQTT
    client = create_mqtt_client(args.broker, args.port, args.username, args.password)
    time.sleep(2)  # Chờ kết nối

    # Nếu đã có đủ 48h, gửi ngay lần đầu
    if len(buffer) >= BUFFER_SIZE:
        csv_str = buffer_to_csv_string(buffer)
        publish_data(client, args.topic, csv_str)

    # Main loop
    try:
        while True:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Đọc cảm biến...")

            # Đọc dữ liệu
            if args.simulate:
                reading = simulate_sensor_reading()
            else:
                reading = read_real_sensors()

            print(f"  Temp={reading['temperature']}°C, "
                  f"Hum={reading['humidity']}%, "
                  f"Rain={reading['precipitation']}mm")

            # Cập nhật buffer
            buffer = append_reading(buffer, reading)
            save_buffer(buffer)

            # Publish nếu đủ data
            if len(buffer) >= BUFFER_SIZE:
                csv_str = buffer_to_csv_string(buffer)
                publish_data(client, args.topic, csv_str)
            else:
                print(f"  [INFO] Buffer mới {len(buffer)}/{BUFFER_SIZE} — chưa đủ để gửi")

            # Chờ chu kỳ tiếp
            print(f"  Chờ {args.interval}s cho chu kỳ tiếp theo...")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[INFO] Dừng publisher...")
    finally:
        client.loop_stop()
        client.disconnect()
        print("[INFO] Đã ngắt kết nối MQTT")


if __name__ == '__main__':
    main()
