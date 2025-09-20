
## python prepare_inference_data.py --input ../elder_care_dataset.json --output inference_input.json

# prepare_inference_data.py
import json
import sys
import argparse
from typing import Any, Dict, List

def extract_features(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    从原始数据中提取推理所需字段，并转换格式
    """
    vitals = raw_data.get("vitals", {})
    behavior = raw_data.get("behavior", {})

    # 解析血压
    bp_str = vitals.get("blood_pressure", "0/0")
    try:
        systolic_bp, diastolic_bp = map(int, bp_str.split('/'))
    except:
        systolic_bp, diastolic_bp = 0, 0  # 默认值

    # 转换风险标志为 0/1
    is_voice_slow = 1.0 if raw_data.get("voice_risk") == "语速慢" else 0.0
    is_night_walking = 1.0 if raw_data.get("image_risk") == "夜间行走频繁" else 0.0

    return {
        "user_id": raw_data["user_id"],
        "age": raw_data["age"],
        "bmi": raw_data["bmi"],
        "heart_rate": vitals.get("heart_rate"),
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "o2_saturation": vitals.get("o2_saturation"),
        "steps_today": behavior.get("steps_today"),
        "inactivity_duration": behavior.get("inactivity_duration"),
        "egfr": raw_data["lab_report"].get("egfr"),
        "is_voice_slow": is_voice_slow,
        "is_night_walking": is_night_walking
    }

def main():
    parser = argparse.ArgumentParser(description="将原始老人健康数据转换为模型推理输入格式")
    parser.add_argument("--input", type=str, required=True, help="原始数据输入文件路径（JSON）")
    parser.add_argument("--output", type=str, default="inference_input.json", help="推理输入数据输出文件路径")

    args = parser.parse_args()

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        print(json.dumps({"error": f"读取输入文件失败: {str(e)}"}, ensure_ascii=False))
        sys.exit(1)

    if isinstance(raw_data, dict):
        raw_data = [raw_data]
    elif not isinstance(raw_data, list):
        print(json.dumps({"error": "JSON 文件必须是一个对象或对象数组"}, ensure_ascii=False))
        sys.exit(1)

    processed_data = []
    for item in raw_data:
        try:
            extracted = extract_features(item)
            processed_data.append(extracted)
        except Exception as e:
            user_id = item.get("user_id", "unknown")
            print(f"⚠️  跳过用户 {user_id}: {str(e)}", file=sys.stderr)

    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 数据转换完成，共处理 {len(processed_data)} 个用户，已保存至: {args.output}", file=sys.stderr)
    except Exception as e:
        print(json.dumps({"error": f"保存输出文件失败: {str(e)}"}, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    main()