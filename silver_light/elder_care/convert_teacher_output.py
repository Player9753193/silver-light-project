## cmd example
##python convert_teacher_output.py \
##  --input ../elder_care_teachered_softlabels.json \
##  --output ./student_input.json

# convert_teacher_output.py
import json
import argparse

def convert_teacher_output(input_json_file, output_jsonl_file):
    # 读取JSON
    with open(input_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"🔍 共加载 {len(data)} 个样本")

    with open(output_jsonl_file, 'w', encoding='utf-8') as out_f:
        for item in data:
            try:
                user_id = item["user_id"]

                s = item["structured_input"]

                try:
                    teacher_ans = json.loads(item["teacher_answer"])
                    soft_labels = teacher_ans["soft_label_scores"]
                except:
                    print(f"❌ 无法解析 teacher_answer: {item['user_id']}")
                    continue

                is_voice_slow = 1 if s["voice_risk"] == "语速慢" else 0
                is_night_walking = 1 if s["image_risk"] == "夜间行走频繁" else 0

                bp_parts = s["vitals"]["blood_pressure"].split("/")
                systolic_bp = int(bp_parts[0])
                diastolic_bp = int(bp_parts[1])

                sample = {
                    "user_id": user_id,
                    "age": s["age"],
                    "bmi": s["bmi"],
                    "heart_rate": s["vitals"]["heart_rate"],
                    "systolic_bp": systolic_bp,
                    "diastolic_bp": diastolic_bp,
                    "o2_saturation": s["vitals"]["o2_saturation"],
                    "steps_today": s["behavior"]["steps_today"],
                    "inactivity_duration": s["behavior"]["inactivity_duration"],
                    "egfr": s["lab_report"]["egfr"],
                    "is_voice_slow": is_voice_slow,
                    "is_night_walking": is_night_walking,
                    "teacher_output": [
                        soft_labels["正常"],
                        soft_labels["疲劳"],
                        soft_labels["心率异常"],
                        soft_labels["血压偏高"],
                        soft_labels["血氧偏低"],
                        soft_labels["情绪低落"],
                        soft_labels["跌倒风险中"],
                        soft_labels["认知模糊早期征兆"]
                    ]
                }

                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"⚠️ 处理 {user_id} 失败: {e}")
                continue

    print(f"✅ 转换完成！已保存为: {output_jsonl_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="原始教师输出 JSON 文件")
    parser.add_argument("--output", type=str, default="input.json", help="转换后的 JSON Lines 文件")
    args = parser.parse_args()

    convert_teacher_output(args.input, args.output)