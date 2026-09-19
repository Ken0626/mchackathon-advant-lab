import json
from jsonschema import validate, ValidationError

# 定義我們的資料契約規則
schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "site": {"type": "integer"},
            "test_number": {"type": "string"},
            "test_suite": {"type": "string"},
            "value": {"type": "number"},
            "low_limit": {"type": "number"},
            "high_limit": {"type": "number"},
            "is_anomaly": {"type": "boolean"},
            "z_score": {"type": ["number", "null"]},
            "anomaly_type": {"type": ["string", "null"]}
        },
        "required": ["site", "value", "is_anomaly"]   # 這幾個欄位一定要有,其他可以缺
    }
}

# 載入你的資料檔,驗證格式
with open("mock_results.json", "r") as f:
    data = json.load(f)

try:
    validate(instance=data, schema=schema)
    print("✅ 格式正確,符合資料契約!")
except ValidationError as e:
    print("❌ 格式有問題:")
    print(e.message)