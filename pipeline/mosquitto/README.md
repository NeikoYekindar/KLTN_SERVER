# Weather Forecast Pipeline

## Kiến trúc hệ thống

```
Raspberry Pi 5          MQTT Broker          Cloud Server              Unity
┌─────────────┐      ┌────────────┐      ┌──────────────────┐      ┌────────────┐
│ Sensors     │      │ Mosquitto  │      │ mqtt_listener    │      │ Digital    │
│ → CSV 48h   │─MQTT─│ topic:     │─────▶│ → last_48h.csv   │      │ Twin 3D    │
│ → publish   │      │ weather/   │      │ → inference.py   │◀─GET─│ poll /api/ │
│             │      │ data       │      │ → forecast.json  │      │ forecast   │
└─────────────┘      └────────────┘      │ → Flask API :5000│      └────────────┘
                                         └──────────────────┘
```

## Cấu trúc thư mục

```
pipeline/
├── edge/
│   └── edge_publisher.py     # Chạy trên Raspberry Pi
├── server/
│   ├── server.py             # MQTT subscriber + Flask API
│   └── inference.py          # Copy file inference.py của bạn vào đây
├── unity/
│   └── WeatherForecastClient.cs  # Script C# cho Unity
├── models/                   # Đặt model files vào đây
│   ├── tcn_model.pth
│   ├── arima_model.pkl
│   └── ensemble_metrics.json (optional)
├── data/                     # Auto-generated
│   ├── last_48h.csv
│   ├── forecast_result.json
│   └── history/
├── mosquitto/config/
│   └── mosquitto.conf
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── test_pipeline.py
```

## Quick Start

### Bước 1: Setup Server

```bash
# Clone/copy project
cd pipeline

# Copy inference.py và models vào đúng chỗ
cp /path/to/inference.py server/
cp /path/to/tcn_model.pth models/
cp /path/to/arima_model.pkl models/

# Cài dependencies
pip install -r requirements.txt
```

### Bước 2: Chạy bằng Docker (recommended)

```bash
# Chạy MQTT broker + Server
docker-compose up -d

# Xem logs
docker-compose logs -f forecast_server
```

### Bước 2 (alt): Chạy thủ công

```bash
# Terminal 1: MQTT broker
docker run -d -p 1883:1883 -v ./mosquitto/config:/mosquitto/config eclipse-mosquitto:2

# Terminal 2: Server
python server/server.py \
    --broker localhost \
    --tcn models/tcn_model.pth \
    --arima models/arima_model.pkl \
    --api-port 5000
```

### Bước 3: Test pipeline

```bash
# Tạo CSV giả lập + upload + kiểm tra
python test_pipeline.py --server http://localhost:5000
```

### Bước 4: Chạy Edge (Raspberry Pi)

```bash
# Trên Raspberry Pi:
pip install paho-mqtt numpy

# Mode mô phỏng (test)
python edge/edge_publisher.py \
    --broker <server-ip> \
    --simulate \
    --init-buffer \
    --interval 60

# Mode cảm biến thật
python edge/edge_publisher.py \
    --broker <server-ip> \
    --interval 3600
```

### Bước 5: Unity

1. Copy `unity/WeatherForecastClient.cs` vào Unity project
2. Tạo Empty GameObject → gắn script
3. Set `serverUrl = "http://<server-ip>:5000"`
4. Trong script khác, subscribe event:

```csharp
void OnEnable()
{
    WeatherForecastClient.Instance.OnForecastUpdated += OnNewForecast;
}

void OnNewForecast(ForecastResult forecast)
{
    var next = forecast.forecast[0];
    Debug.Log($"Temp h+1: {next.temperature}°C, Rain: {next.precipitation}mm");

    // Cập nhật Digital Twin scene ở đây
    UpdateWeatherVisualization(next);
}
```

## API Endpoints

| Method | Endpoint              | Mô tả                          |
|--------|-----------------------|---------------------------------|
| GET    | `/api/forecast`       | Toàn bộ forecast 6h            |
| GET    | `/api/forecast/latest`| Chỉ bước h+1                   |
| GET    | `/api/forecast/step/3`| Bước cụ thể (1-6)              |
| GET    | `/api/current`        | Dữ liệu sensor mới nhất        |
| GET    | `/api/status`         | Health check                    |
| POST   | `/api/trigger`        | Trigger inference thủ công      |
| POST   | `/api/upload_csv`     | Upload CSV (thay MQTT khi test) |

## Ví dụ response `/api/forecast`

```json
{
  "generated_at": "2026-04-04T15:00:00",
  "based_on_data_until": "2026-04-04T14:00:00",
  "horizon_hours": 6,
  "model_used": "TCN+ARIMA ensemble",
  "forecast": [
    {
      "timestamp": "2026-04-04T15:00:00",
      "horizon_step": 1,
      "temperature": 31.25,
      "humidity": 72.8,
      "wind_speed": 12.5,
      "precipitation": 0.02,
      "pressure": 1008.5,
      "...": "..."
    }
  ]
}
```

## Luồng dữ liệu chi tiết

1. **Raspberry Pi** đọc sensor mỗi giờ → append vào buffer 48 hàng
2. Buffer đầy → serialize thành CSV string → publish MQTT topic `weather/data`
3. **Server** subscriber nhận message → parse JSON → lưu `last_48h.csv`
4. Server gọi `inference.py --csv last_48h.csv` → output `forecast_result.json`
5. Flask API serve JSON tại `/api/forecast`
6. **Unity** poll GET mỗi 30s → parse JSON → cập nhật Digital Twin scene




python3 server/server.py     --broker localhost     --mqtt-port 1883     --tcn models/tcn_model.pth     --arima models/arima_model.pkl     --inference-script server/inference.py     --ensemble_json models/ensemble_stacking_metrics.json     --api-port 5000


python3 server/server.py \
    --broker localhost \
    --tcn models/tcn_model.pth \
    --lstm models/lstm_model.pth \
    --arima models/arima_model.pkl \
    --ensemble_json_tcn models/ensemble_stacking_metrics.json \
    --ensemble_json_lstm models/ensemble_lstm_SGN_metrics.json \
    --inference-script-tcn server/inference.py \
    --inference-script-lstm server/inference_lstm_arima.py \
    --api-port 5000

python3 server/server_dual_tcn.py  --broker   localhost --tcn      models/tcn_model_24h.pth   --tcn_hard models/tcn_hard_model_24h.pth     --api-port 5000

    # ===== TCN + ARIMA =====
curl.exe http://192.168.25.129:5000/api/tcn/forecast
curl.exe http://192.168.25.129:5000/api/tcn/forecast/latest
curl.exe http://192.168.25.129:5000/api/tcn/forecast/step/1

# ===== LSTM + ARIMA =====
curl.exe http://192.168.25.129:5000/api/lstm/forecast
curl.exe http://192.168.25.129:5000/api/lstm/forecast/latest
curl.exe http://192.168.25.129:5000/api/lstm/forecast/step/1

# ===== So sánh 2 model =====
curl.exe http://192.168.25.129:5000/api/compare

# ===== Chung =====
curl.exe http://192.168.25.129:5000/api/current
curl.exe http://192.168.25.129:5000/api/status
curl.exe http://192.168.25.129:5000/api/forecast          # backward compatible = TCN
curl.exe -X POST -F "file=@test.csv" http://192.168.25.129:5000/api/upload_csv
curl.exe -X POST http://192.168.25.129:5000/api/trigger
