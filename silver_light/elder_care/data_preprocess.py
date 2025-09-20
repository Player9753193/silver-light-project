
## cmd example
##python data_preprocess.py \
##  --input ../elder_care_teachered_softlabels.json \
##  --output ./output.json

import json
import numpy as np
import argparse
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import codecs

# 固定特征顺序
FEATURE_NAMES = [
    'age', 'bmi', 'heart_rate', 'systolic_bp', 'diastolic_bp',
    'o2_saturation', 'steps_today', 'inactivity_duration',
    'egfr', 'is_voice_slow', 'is_night_walking'
]

# 软标签类别
STATE_LABELS = [
    "正常", "疲劳", "心率异常", "血压偏高",
    "血氧偏低", "情绪低落", "跌倒风险中", "认知模糊早期征兆"
]


def load_json_data(file_path):
    """健壮加载 JSON 数据，支持：每行一个 JSON"""
    data = []
    errors = 0

    # 使用 codecs 避免 BOM 问题
    with codecs.open(file_path, 'r', encoding='utf-8-sig') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
                data.append(sample)

            except json.JSONDecodeError as e:
                if '}' in line and '{' in line[1:]:
                    print(f"🟡 第 {line_num} 行疑似多个 JSON 拼接，尝试分割：{line[:50]}...")
                    parts = line.split('}{')
                    for i, part in enumerate(parts):
                        if i == 0:
                            part += '}'
                        elif i == len(parts) - 1:
                            part = '{' + part
                        else:
                            part = '{' + part + '}'

                        try:
                            sample = json.loads(part)
                            data.append(sample)
                        except:
                            errors += 1
                            continue
                else:
                    errors += 1
                    if errors < 10:
                        print(f"❌ 第 {line_num} 行 JSON 解析错误: {e}，跳过: {line[:50]}")
    print(f"✅ 成功加载 {len(data)} 条样本，跳过 {errors} 条错误")
    return data


def extract_features_and_labels(data):
    """提取特征和软标签"""
    X_list = []
    Y_list = []

    for i, sample in enumerate(data):
        try:
            x = [sample[feat] for feat in FEATURE_NAMES]
            X_list.append(x)

            if isinstance(sample.get('teacher_output'), list):
                y = sample['teacher_output']
            elif isinstance(sample.get('teacher_output'), dict):
                y = [sample['teacher_output'].get(state, 0.0) for state in STATE_LABELS]
            else:
                y = [0.0] * len(STATE_LABELS)
            Y_list.append(y)

        except Exception as e:
            print(f"⚠️ 处理第 {i+1} 个样本失败: {e}，数据: {sample}")
            continue

    return np.array(X_list), np.array(Y_list)


def normalize_features(X, method='standard'):
    """归一化特征"""
    if X.size == 0:
        raise ValueError("❌ 归一化失败：X 为空")

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        return X, {"method": "none"}

    X_scaled = scaler.fit_transform(X)

    scaler_info = {
        "method": method,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "min": getattr(scaler, 'data_min_', None).tolist() if method == 'minmax' else None,
        "max": getattr(scaler, 'data_max_', None).tolist() if method == 'minmax' else None
    }

    return X_scaled, scaler_info


def main(input_file, output_dir, normalize_method):
    os.makedirs(output_dir, exist_ok=True)

    print("🔍 正在加载数据...")
    data = load_json_data(input_file)

    if len(data) == 0:
        raise ValueError("❌ 数据加载失败：没有有效样本，请检查输入文件格式！")

    print("📊 正在提取特征和软标签...")
    X, Y = extract_features_and_labels(data)

    if X.shape[0] == 0:
        raise ValueError("❌ 特征提取失败：没有有效样本，请检查字段名是否匹配！")

    print(f"✅ 构建输入特征矩阵: {X.shape}")
    print(f"✅ 构建软标签矩阵: {Y.shape}")

    if normalize_method and normalize_method != 'none':
        print(f"🔄 正在使用 {normalize_method} 方法归一化特征...")
        X, scaler_info = normalize_features(X, method=normalize_method)
        with open(f"{output_dir}/scaler_info.json", 'w', encoding='utf-8') as f:
            json.dump(scaler_info, f, ensure_ascii=False, indent=2)
        print(f"✅ 归一化参数已保存: {output_dir}/scaler_info.json")
    else:
        scaler_info = {"method": "none"}

    np.save(f"{output_dir}/X_features.npy", X)
    np.save(f"{output_dir}/Y_soft_labels.npy", Y)
    print(f"✅ 特征矩阵已保存: {output_dir}/X_features.npy")
    print(f"✅ 软标签矩阵已保存: {output_dir}/Y_soft_labels.npy")

    with open(f"{output_dir}/feature_names.json", 'w', encoding='utf-8') as f:
        json.dump(FEATURE_NAMES, f, ensure_ascii=False, indent=2)

    print("🎉 数据预处理完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="康养知识蒸馏数据预处理")
    parser.add_argument("--input", type=str, required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output_dir", type=str, default="./student_data", help="输出目录")
    parser.add_argument("--normalize", type=str, default="standard",
                        choices=["standard", "minmax", "robust", "none"], help="归一化方法")

    args = parser.parse_args()
    main(args.input, args.output_dir, args.normalize if args.normalize != 'none' else None)