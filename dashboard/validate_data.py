import json
from jsonschema import validate, ValidationError

# 資料契約 v2 —— 對應元珍 AnomalyDetector 實際輸出格式
schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "site": {"type": "integer"},
            "test_number": {"type": "string"},
            "value": {"type": "number"},
            "is_anomaly": {"type": "boolean"},
            "anomaly_type": {"type": ["string", "null"]},
            "action": {"type": "string"},
            "reason": {"type": "string"}
        },
        "required": ["site", "value", "is_anomaly", "action"]
    }
}

with open("mock_results.json", "r") as f:
    data = json.load(f)

try:
    validate(instance=data, schema=schema)
    print("✅ 格式正確,符合資料契約 v2!")
except ValidationError as e:
    print("❌ 格式有問題:")
    print(e.message)
