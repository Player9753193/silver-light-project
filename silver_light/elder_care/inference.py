import torch
import json
import sys
import argparse
from student_model import MLPWithAttention
import numpy as np

def load_model(model_path="student_mlp_attn_best.pth", scaler_path="./student_data/scaler_info.json"):
    """加载训练好的模型和归一化参数"""
    try:
        model = MLPWithAttention(input_dim=11, hidden_dim=64, num_classes=8)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        with open(scaler_path, 'r', encoding='utf-8') as f:
            scaler = json.load(f)

        return model, scaler
    except Exception as e:
        print(json.dumps({"error": f"加载模型失败: {str(e)}"}, ensure_ascii=False))
        sys.exit(1)

def preprocess_input(raw_features: np.ndarray, scaler):
    """应用与训练时相同的归一化"""
    mean = np.array(scaler['mean'])
    scale = np.array(scaler['scale'])
    return (raw_features - mean) / scale

def predict_health_status(model, scaler, input_vector):
    """预测单个样本的健康状态概率分布"""
    x = preprocess_input(np.array([input_vector]), scaler)
    x = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    states = [
        "正常", "疲劳", "心率异常", "血压偏高",
        "血氧偏低", "情绪低落", "跌倒风险中", "认知模糊早期征兆"
    ]

    return {state: float(prob) for state, prob in zip(states, probs)}

def load_input_from_file(input_file):
    """从文件加载输入数据，支持单个对象或对象列表"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(json.dumps({"error": f"读取输入文件失败: {str(e)}"}, ensure_ascii=False))
        sys.exit(1)

def save_result_to_file(results, output_file):
    """将结果保存到文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ 预测结果已保存到: {output_file}", file=sys.stderr)
    except Exception as e:
        print(json.dumps({"error": f"保存结果失败: {str(e)}"}, ensure_ascii=False))
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="老年健康状态预测推理脚本（文件输入）")
    parser.add_argument("--input", type=str, required=True, help="输入 JSON 文件路径（包含一个或多个用户数据）")
    parser.add_argument("--output", type=str, default="prediction_results.json", help="输出结果文件路径")
    parser.add_argument("--model-path", type=str, default="./student_data/student_mlp_attn_best.pth", help="模型文件路径")
    parser.add_argument("--scaler-path", type=str, default="./student_data/scaler_info.json", help="归一化参数文件路径")

    args = parser.parse_args()

    # 加载模型和 scaler
    model, scaler = load_model(args.model_path, args.scaler_path)

    # 加载输入数据
    raw_data = load_input_from_file(args.input)

    # 确保数据是列表形式
    if isinstance(raw_data, dict):
        raw_data = [raw_data]
    elif not isinstance(raw_data, list):
        print(json.dumps({"error": "输入文件必须是一个对象或对象数组"}, ensure_ascii=False))
        sys.exit(1)

    # 批量预测
    results = {}
    for item in raw_data:
        user_id = item.get("user_id", "unknown")
        try:
            input_vector = [
                item["age"],
                item["bmi"],
                item["heart_rate"],
                item["systolic_bp"],
                item["diastolic_bp"],
                item["o2_saturation"],
                item["steps_today"],
                item["inactivity_duration"],
                item["egfr"],
                float(item.get("is_voice_slow", 0.0)),
                float(item.get("is_night_walking", 0.0))
            ]
            result = predict_health_status(model, scaler, input_vector)
            results[user_id] = result
        except KeyError as e:
            results[user_id] = {"error": f"缺少字段: {str(e)}"}
        except Exception as e:
            results[user_id] = {"error": f"预测失败: {str(e)}"}

    save_result_to_file(results, args.output)

if __name__ == "__main__":
    main()