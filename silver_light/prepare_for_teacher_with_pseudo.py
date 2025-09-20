import json
import sys
from typing import Dict, List

# 配置：健康状态类别
HEALTH_CATEGORIES = [
    "正常",
    "疲劳",
    "心率异常",
    "血压偏高",
    "血氧偏低",
    "情绪低落",
    "跌倒风险中",
    "认知模糊早期征兆"
]



def build_teacher_prompt_for_softlabel(structured_input):
    # 拆分血压
    bp = structured_input["vitals"]["blood_pressure"]
    systolic, diastolic = bp.split("/")

    prompt = f"""你是一位资深老年健康评估专家，请根据以下多模态信息，对老人当前健康状态进行综合判断。

### 一、基本信息
- 年龄：{structured_input['age']}岁
- 性别：{structured_input['gender']}
- 慢性病史：{"、".join(structured_input['chronic_diseases']) if structured_input['chronic_diseases'] else "无已知慢性病"}
- 身高：{structured_input['height']}米，体重：{structured_input['weight']}公斤，BMI：{structured_input['bmi']}

### 二、生理指标
- 心率：{structured_input['vitals']['heart_rate']} 次/分钟
- 血压：{bp} mmHg
- 血氧饱和度：{structured_input['vitals']['o2_saturation']}%
- 体温：{structured_input['vitals']['temperature']}°C

### 三、行为数据
- 今日步数：{structured_input['behavior']['steps_today']} 步
- 夜间起夜次数：{structured_input['behavior']['night_awakenings']} 次
- 久坐时长：{structured_input['behavior']['inactivity_duration']} 小时
- 主要活动：{structured_input['behavior']['primary_activity']}

### 四、语音分析
- 风险特征：{structured_input['voice_risk']}

### 五、图像行为识别
- 动作风险：{structured_input['image_risk']}

### 六、近期检验报告（{structured_input['lab_report']['date']}）
- 空腹血糖：{structured_input['lab_report']['glucose_fasting']} mmol/L
- 糖化血红蛋白：{structured_input['lab_report']['hba1c']}%
- 肌酐：{structured_input['lab_report']['creatinine']} μmol/L
- 肾小球滤过率：{structured_input['lab_report']['egfr']} mL/min
- 医生备注：{structured_input['lab_report']['notes']}

---

### 请完成以下任务：

1. **状态可能性评分**
   请为以下 8 种健康状态分别评估其可能性，使用 **0~100 分制**（分数越高表示越可能），允许多项高分，总分不必为100：
   正常、疲劳、心率异常、血压偏高、血氧偏低、情绪低落、跌倒风险中、认知模糊早期征兆

   请严格按照以下格式输出评分：
   正常：__分
   疲劳：__分
   心率异常：__分
   血压偏高：__分
   血氧偏低：__分
   情绪低落：__分
   跌倒风险中：__分
   认知模糊早期征兆：__分

2. **生成个性化护理建议**
   用亲切、口语化的中文写一段建议（60字左右），包含具体可操作动作。

---

### 输出格式要求：
请以标准 JSON 格式输出，不要任何额外解释或 Markdown 符号。
只输出一个 JSON 对象，包含两个字段：
- "soft_label_scores": 一个字典，键为状态名称，值为整数分数（0~100）
- "care_advice": 一个字符串，为护理建议

示例结构如下（不要复制此内容，仅参考格式）：
{{
  "soft_label_scores": {{
    "正常": 0,
    "疲劳": 0,
    "心率异常": 0,
    "血压偏高": 0,
    "血氧偏低": 0,
    "情绪低落": 0,
    "跌倒风险中": 0,
    "认知模糊早期征兆": 0
  }},
  "care_advice": ""
}}
"""

    return prompt.strip()

def convert_to_teacher_input_with_pseudo(input_file: str, output_file: str):
    """
    主函数：读取原始模拟数据集，保留伪标签，生成教师模型可用的 Prompt
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"✅ 成功加载 {len(dataset)} 条模拟数据（含伪教师标签）")

        processed_data = []
        for i, record in enumerate(dataset):
            pseudo_status = record.get("teacher_health_status", "未知")
            pseudo_advice = record.get("teacher_care_advice", "无建议")

            prompt = build_teacher_prompt_for_softlabel(record)

            processed_item = {
                "user_id": record.get("user_id", f"unknown_{i}"),
                "pseudo_teacher_labels": {
                    "health_status": pseudo_status,
                    "care_advice": pseudo_advice
                },
                "structured_input": record, 
                "teacher_input_prompt": prompt
            }
            processed_data.append(processed_item)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        print(f"✅ 数据转换完成！")
        print(f"📁 输出文件：{output_file}")
        print(f"📊 总条数：{len(processed_data)}")
        print(f"💡 提示：每条数据均保留伪标签，并生成可用于生成软标签的 Prompt")

    except FileNotFoundError:
        print(f"❌ 错误：找不到输入文件 '{input_file}'，请确认文件路径正确")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 处理过程中发生错误：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # 设置文件路径
    INPUT_JSON = "elder_care_dataset.json"        
    OUTPUT_JSON = "elder_care_for_teacher_with_pseudo.json" 
    
    # 执行转换
    convert_to_teacher_input_with_pseudo(INPUT_JSON, OUTPUT_JSON)