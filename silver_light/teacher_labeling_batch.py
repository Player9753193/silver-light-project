#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import time
import os
import random
import sys
from typing import List, Dict, Any
from http import HTTPStatus
import dashscope
from dashscope import Generation
from tqdm import tqdm


# ======================
# 🛠️ 配置区
# ======================

# ✅ 请根据你的实际情况修改
DASHSCOPE_API_KEY = "sk-xxxxxxxxxxxxxxxx" 
INPUT_FILE = "elder_care_for_teacher_with_pseudo.json"
OUTPUT_FILE = "elder_care_teachered_softlabels.json"

# 模型：qwen-max（最强）、qwen-plus（平衡）、qwen-turbo（最快）
QWEN_MODEL = "qwen-max"
TEMPERATURE = 0.2
TOP_P = 0.8
MAX_TOKENS = 800
RETRY_TIMES = 3
SLEEP_BETWEEN_CALLS = 1.0



def init_dashscope():
    dashscope.api_key = DASHSCOPE_API_KEY

def load_dataset(file_path: str) -> List[Dict]:
    """自动识别 JSON 或 JSONL 格式"""
    _, ext = os.path.splitext(file_path)
    dataset = []

    with open(file_path, 'r', encoding='utf-8') as f:
        if ext.lower() == '.jsonl':
            for line in f:
                line = line.strip()
                if line:
                    try:
                        dataset.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSONL 解析失败: {e}")
        else:  # .json
            try:
                data = json.load(f)
                if isinstance(data, list):
                    dataset = data
                else:
                    dataset = [data]
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON 解析失败: {e}")
    return dataset

def save_dataset(dataset: List[Dict], file_path: str):
    _, ext = os.path.splitext(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            if ext.lower() == '.jsonl':
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
                break  # 只保存一次

def call_qwen3(prompt: str) -> str:
    """调用 Qwen3，仅关注调用成功与否，不解析内容"""
    for attempt in range(RETRY_TIMES):
        try:
            response = Generation.call(
                model=QWEN_MODEL,
                prompt=prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                result_format='message'
            )

            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0]['message']['content'].strip()
            else:
                print(f"❌ API 错误 [尝试 {attempt+1}/{RETRY_TIMES}]: "
                      f"状态码={response.status_code}, 错误码={response.code}, 信息={response.message}")
                time.sleep(2 ** attempt + random.uniform(0, 1))

        except Exception as e:
            print(f"⚠️  调用异常 [尝试 {attempt+1}/{RETRY_TIMES}]: {str(e)}")
            time.sleep(2 ** attempt + random.uniform(0, 1))

    return "[Qwen3 生成失败]"


def main():
    print("🚀 开始生成 Qwen3 教师模型软标签...")

    init_dashscope()

    print(f"📁 正在加载数据集: {INPUT_FILE}")
    samples = load_dataset(INPUT_FILE)
    print(f"✅ 加载完成，共 {len(samples)} 个样本")

    to_process = [
        s for s in samples
        if not s.get("teacher_answer") or s["teacher_answer"].startswith("[Qwen3 生成失败]")
    ]
    print(f"🔍 发现 {len(to_process)} 个待生成软标签的样本")

    if len(to_process) == 0:
        print("✅ 所有样本均已生成答案，无需处理。")
        return

    # 批量生成
    success_count = 0
    for sample in tqdm(to_process, desc="🧠 生成中", unit="sample"):
        prompt_field = "student_model_input" if "student_model_input" in sample else "teacher_input_prompt"
        prompt = sample.get(prompt_field, "").strip()

        if not prompt:
            sample["teacher_answer"] = "[错误：无有效 prompt]"
            continue

        answer = call_qwen3(prompt)
        sample["teacher_answer"] = answer
        sample["model_used"] = QWEN_MODEL
        sample["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        if not answer.startswith("[Qwen3 生成失败]"):
            success_count += 1

        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\n💾 正在保存结果到: {OUTPUT_FILE}")
    save_dataset(samples, OUTPUT_FILE)

    total = len(samples)
    failed_count = len(to_process) - success_count
    print(f"""
✅ 软标签生成完成！
📊 统计结果：
   - 总样本数: {total}
   - 成功生成: {success_count}
   - 失败数量: {failed_count}
   - 输出文件: {OUTPUT_FILE}
💡 提示：该文件可直接用于知识蒸馏训练（如 LLaMA-Factory）
    """)

if __name__ == "__main__":
    main()