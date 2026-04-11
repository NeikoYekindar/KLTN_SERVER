/*
=============================================================================
WeatherForecastClient.cs — Unity Script
=============================================================================
Gắn vào một GameObject trong Unity scene.
Poll API server mỗi N giây để lấy dữ liệu dự báo.

Sử dụng:
  1. Tạo Empty GameObject, đặt tên "ForecastManager"
  2. Kéo thả script này vào
  3. Set serverUrl = "http://<server-ip>:5000"
  4. Các script khác lấy dữ liệu qua: ForecastManager.Instance.LatestForecast
=============================================================================
*/

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Data classes — mapping từ forecast_result.json
/// </summary>

[Serializable]
public class ForecastStep
{
    public string timestamp;
    public int horizon_step;
    public float temperature;
    public float feels_like;
    public float dewpoint;
    public float humidity;
    public float wind_speed;
    public float gust_speed;
    public float wind_direction;
    public float pressure;
    public float precipitation;
    public float rain_probability;
    public float uv_index;
    public float visibility;
    public float cloud;
}

[Serializable]
public class ForecastWeights
{
    // Trọng số dưới dạng dictionary — parse thủ công nếu cần
    public string tcn;
    public string arima;
}

[Serializable]
public class ForecastResult
{
    public string generated_at;
    public string based_on_data_until;
    public int horizon_hours;
    public string model_used;
    public string[] target_cols;
    public ForecastStep[] forecast;
}

[Serializable]
public class SensorData
{
    public string device_id;
    public string received_at;
    public int num_rows;
    public string[] columns;
}

[Serializable]
public class ServerStatus
{
    public string status;
    public string started_at;
    public string last_data_received;
    public string last_inference_run;
    public int inference_count;
    public bool has_forecast;
    public bool has_sensor_data;
}


/// <summary>
/// Main client — Singleton pattern
/// </summary>
public class WeatherForecastClient : MonoBehaviour
{
    // ========== Singleton ==========
    public static WeatherForecastClient Instance { get; private set; }

    // ========== Inspector Settings ==========
    [Header("Server Connection")]
    [Tooltip("URL của Flask server, vd: http://192.168.1.100:5000")]
    public string serverUrl = "http://localhost:5000";

    [Tooltip("Tần suất poll API (giây)")]
    [Range(5, 300)]
    public float pollInterval = 30f;

    [Tooltip("Tự động bắt đầu poll khi Start()")]
    public bool autoStart = true;

    [Header("Debug")]
    public bool logResponses = false;

    // ========== Public Data ==========
    /// <summary>Kết quả dự báo mới nhất</summary>
    public ForecastResult LatestForecast { get; private set; }

    /// <summary>Dữ liệu cảm biến mới nhất</summary>
    public SensorData LatestSensorData { get; private set; }

    /// <summary>Server có đang online không</summary>
    public bool IsConnected { get; private set; }

    /// <summary>Thời điểm nhận forecast cuối cùng</summary>
    public DateTime LastFetchTime { get; private set; }

    // ========== Events — các script khác subscribe ==========
    /// <summary>Gọi khi có forecast mới</summary>
    public event Action<ForecastResult> OnForecastUpdated;

    /// <summary>Gọi khi có sensor data mới</summary>
    public event Action<SensorData> OnSensorDataUpdated;

    /// <summary>Gọi khi mất/có kết nối</summary>
    public event Action<bool> OnConnectionChanged;

    // ========== Private ==========
    private Coroutine _pollCoroutine;
    private string _lastForecastHash = "";


    void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        DontDestroyOnLoad(gameObject);
    }

    void Start()
    {
        if (autoStart)
            StartPolling();
    }

    // ============================================================
    // Public API
    // ============================================================

    /// <summary>Bắt đầu poll server</summary>
    public void StartPolling()
    {
        if (_pollCoroutine != null)
            StopCoroutine(_pollCoroutine);
        _pollCoroutine = StartCoroutine(PollLoop());
        Debug.Log($"[Forecast] Polling started: {serverUrl} every {pollInterval}s");
    }

    /// <summary>Dừng poll</summary>
    public void StopPolling()
    {
        if (_pollCoroutine != null)
        {
            StopCoroutine(_pollCoroutine);
            _pollCoroutine = null;
        }
        Debug.Log("[Forecast] Polling stopped");
    }

    /// <summary>Fetch một lần ngay lập tức</summary>
    public void FetchNow()
    {
        StartCoroutine(FetchForecast());
    }

    /// <summary>Lấy dự báo tại step cụ thể (1-6)</summary>
    public ForecastStep GetForecastStep(int step)
    {
        if (LatestForecast?.forecast == null) return null;
        foreach (var f in LatestForecast.forecast)
        {
            if (f.horizon_step == step) return f;
        }
        return null;
    }

    /// <summary>Lấy step gần nhất (h+1)</summary>
    public ForecastStep GetNextHourForecast()
    {
        return GetForecastStep(1);
    }

    /// <summary>Kiểm tra có mưa trong N giờ tới không</summary>
    public bool WillRainInNextHours(int hours, float threshold = 0.5f)
    {
        if (LatestForecast?.forecast == null) return false;
        foreach (var f in LatestForecast.forecast)
        {
            if (f.horizon_step <= hours && f.precipitation > threshold)
                return true;
        }
        return false;
    }

    /// <summary>Lấy nhiệt độ max trong dự báo</summary>
    public float GetMaxTemperature()
    {
        if (LatestForecast?.forecast == null) return float.NaN;
        float max = float.MinValue;
        foreach (var f in LatestForecast.forecast)
            if (f.temperature > max) max = f.temperature;
        return max;
    }

    // ============================================================
    // Coroutines
    // ============================================================

    private IEnumerator PollLoop()
    {
        while (true)
        {
            yield return FetchForecast();
            yield return FetchSensorData();
            yield return new WaitForSeconds(pollInterval);
        }
    }

    private IEnumerator FetchForecast()
    {
        string url = $"{serverUrl}/api/forecast";

        using (UnityWebRequest req = UnityWebRequest.Get(url))
        {
            req.timeout = 10;
            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
            {
                string json = req.downloadHandler.text;

                if (logResponses)
                    Debug.Log($"[Forecast] Response: {json.Substring(0, Mathf.Min(200, json.Length))}...");

                // Chỉ update nếu data thay đổi
                string hash = json.GetHashCode().ToString();
                if (hash != _lastForecastHash)
                {
                    try
                    {
                        ForecastResult result = JsonUtility.FromJson<ForecastResult>(json);
                        LatestForecast = result;
                        LastFetchTime = DateTime.Now;
                        _lastForecastHash = hash;

                        Debug.Log($"[Forecast] Updated: {result.horizon_hours}h forecast, " +
                                  $"model={result.model_used}, " +
                                  $"steps={result.forecast?.Length ?? 0}");

                        OnForecastUpdated?.Invoke(result);
                    }
                    catch (Exception e)
                    {
                        Debug.LogError($"[Forecast] Parse error: {e.Message}");
                    }
                }

                SetConnected(true);
            }
            else
            {
                Debug.LogWarning($"[Forecast] Fetch failed: {req.error}");
                SetConnected(false);
            }
        }
    }

    private IEnumerator FetchSensorData()
    {
        string url = $"{serverUrl}/api/current";

        using (UnityWebRequest req = UnityWebRequest.Get(url))
        {
            req.timeout = 10;
            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
            {
                try
                {
                    SensorData data = JsonUtility.FromJson<SensorData>(req.downloadHandler.text);
                    LatestSensorData = data;
                    OnSensorDataUpdated?.Invoke(data);
                }
                catch (Exception e)
                {
                    Debug.LogError($"[Sensor] Parse error: {e.Message}");
                }
            }
        }
    }

    private void SetConnected(bool connected)
    {
        if (IsConnected != connected)
        {
            IsConnected = connected;
            OnConnectionChanged?.Invoke(connected);
            Debug.Log($"[Forecast] Connection: {(connected ? "ONLINE" : "OFFLINE")}");
        }
    }
}


/*
=============================================================================
Ví dụ sử dụng trong script khác:
=============================================================================

public class WeatherVisualizer : MonoBehaviour
{
    void OnEnable()
    {
        WeatherForecastClient.Instance.OnForecastUpdated += HandleNewForecast;
    }

    void OnDisable()
    {
        if (WeatherForecastClient.Instance != null)
            WeatherForecastClient.Instance.OnForecastUpdated -= HandleNewForecast;
    }

    void HandleNewForecast(ForecastResult forecast)
    {
        // Cập nhật Digital Twin
        var nextHour = forecast.forecast[0];

        // Thay đổi thời tiết trong scene
        UpdateRain(nextHour.precipitation);
        UpdateWind(nextHour.wind_speed, nextHour.wind_direction);
        UpdateTemperatureDisplay(nextHour.temperature);
        UpdateCloudCover(nextHour.cloud);

        // Kiểm tra cảnh báo
        if (nextHour.precipitation > 10f)
            ShowFloodWarning();

        if (WeatherForecastClient.Instance.WillRainInNextHours(3))
            ShowRainAlert();
    }
}

=============================================================================
*/
