import json
import re

def convert_anomaly_results_to_json(alerts, site, value, test_number="220"):
    """
    把 detector.process_new_data() 回傳的 alerts list
    轉換成我們 Dashboard 要的格式。
    就算 alerts 是空的(代表正常),也產生一筆 is_anomaly=false 的記錄。
    """
    if not alerts:
        return {
            "site": site,
            "test_number": test_number,
            "value": value,
            "is_anomaly": False,
            "anomaly_type": None,
            "action": "none",
            "reason": ""
        }
    else:
        alert = alerts[0]
        return {
            "site": site,
            "test_number": test_number,
            "value": value,
            "is_anomaly": alert["is_anomaly"],
            "anomaly_type": alert["anomaly_type"],
            "action": alert["action"],
            "reason": alert["reason"]
        }


def convert_temperature_string_to_json(temp_message):
    """
    把 predict_temperature() 回傳的字串,例如:
    "site[1]: 23.45 site[2]: 24.10 site[3]: 22.80 site[4]: 25.00 "
    轉成結構化的 list。
    """
    pattern = r"site\[(\d+)\]:\s*([\d.]+)"
    matches = re.findall(pattern, temp_message)
    return [{"site": int(site), "predicted_temperature": float(temp)} for site, temp in matches]


if __name__ == "__main__":
    result1 = convert_anomaly_results_to_json([], site=1, value=1.28)
    print(json.dumps(result1, indent=2, ensure_ascii=False))

    fake_alert = [{"is_anomaly": True, "anomaly_type": "Global Feature Anomaly", "action": "set_pause", "reason": "MSE 0.1234 超出閾值"}]
    result2 = convert_anomaly_results_to_json(fake_alert, site=3, value=1.9)
    print(json.dumps(result2, indent=2, ensure_ascii=False))

    fake_temp_msg = "site[1]: 23.45 site[2]: 24.10 site[3]: 22.80 site[4]: 25.00 "
    result3 = convert_temperature_string_to_json(fake_temp_msg)
    print(json.dumps(result3, indent=2, ensure_ascii=False))
