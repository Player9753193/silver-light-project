import json
import random
from datetime import datetime, timedelta

# 设置随机种子（可选）
random.seed(42)

genders = ['男', '女']
chronic_diseases_options = [
    [], 
    ['高血压'], 
    ['糖尿病'], 
    ['冠心病'], 
    ['慢性支气管炎'], 
    ['高血压', '糖尿病'], 
    ['高血压', '冠心病']
]

health_status_options = [
    '正常', 
    '疲劳', 
    '心率异常', 
    '血压偏高', 
    '血氧偏低', 
    '情绪低落', 
    '跌倒风险中', 
    '认知模糊早期征兆'
]

care_advice_templates = {
    '正常': "今日各项指标平稳，继续保持规律作息和适度运动，建议每日步行3000步以上。",
    '疲劳': "近期活动量减少，精神状态偏疲倦，建议增加日间光照 exposure，适当补充维生素B族。",
    '心率异常': "监测到心率波动较大，建议避免剧烈情绪波动，定时测量心率，必要时就医检查。",
    '血压偏高': "收缩压持续高于140，建议低盐饮食，保持情绪稳定，规律服用降压药。",
    '血氧偏低': "血氧饱和度偏低，可能与呼吸功能下降有关，建议保持室内通风，避免长时间卧床。",
    '情绪低落': "语音语调分析显示情绪低落倾向，建议家人多陪伴交流，参与轻度社交活动。",
    '跌倒风险中': "行动迟缓，下肢力量减弱，建议家中加装扶手，穿防滑鞋，避免独自上下楼梯。",
    '认知模糊早期征兆': "言语略显混乱，短期记忆减退，建议进行认知训练，定期神经科随访。"
}

def generate_elder_data():
    age = random.randint(65, 90)
    gender = random.choice(genders)
    chronic_diseases = random.choices(chronic_diseases_options, weights=[0.3, 0.25, 0.15, 0.1, 0.08, 0.07, 0.05])[0]
    
    height = round(random.uniform(1.55, 1.75), 2)
    weight = round(random.uniform(50, 75), 1)
    bmi = round(weight / (height ** 2), 1)
    
    heart_rate = random.randint(60, 100)
    if heart_rate > 90: heart_rate_label = '偏快'
    elif heart_rate < 60: heart_rate_label = '偏慢'
    else: heart_rate_label = '正常'

    bp_systolic = random.randint(120, 180)  # 收缩压
    bp_diastolic = random.randint(70, 100)  # 舒张压
    blood_pressure = f"{bp_systolic}/{bp_diastolic}"
    bp_status = '正常' if bp_systolic < 140 else '偏高'

    o2_saturation = random.randint(90, 100)
    o2_status = '正常' if o2_saturation >= 95 else '偏低'

    temperature = round(random.uniform(36.0, 37.5), 1)
    temp_status = '正常'

    # 行为数据
    steps_today = random.randint(500, 5000)
    night_awakenings = random.randint(0, 5)
    inactivity_duration = round(random.uniform(2.0, 10.0), 1)  # 小时
    primary_activity = random.choice(['居家休息', '室内走动', '看电视', '简单家务'])

    # 语音与图像风险
    voice_risk = random.choice(['正常', '语速慢', '声音微弱', '语调低沉', '言语不清'])
    image_risk = random.choice(['动作正常', '动作迟缓', '站立不稳', '坐卧时间长', '夜间行走频繁'])

    lab_report = {
        "date": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
        "glucose_fasting": round(random.uniform(4.0, 8.0), 1),  # mmol/L
        "hba1c": round(random.uniform(5.0, 8.0), 1),  # 糖化血红蛋白
        "creatinine": round(random.uniform(60, 110), 1),  # umol/L
        "egfr": round(random.uniform(45, 90), 1),  # 肾小球滤过率
        "notes": "血糖控制一般，肾功能轻度下降，建议内分泌科随访。" if '糖尿病' in chronic_diseases else "各项指标基本稳定。"
    }

    # 教师模型“标注” —— 基于规则模拟
    risk_factors = []
    if bp_systolic >= 140: risk_factors.append('血压偏高')
    if heart_rate > 90 or heart_rate < 60: risk_factors.append('心率异常')
    if o2_saturation < 95: risk_factors.append('血氧偏低')
    if steps_today < 1000: risk_factors.append('活动不足')
    if night_awakenings >= 3: risk_factors.append('睡眠中断')
    if inactivity_duration > 7: risk_factors.append('久坐')
    if '语速慢' in voice_risk or '言语不清' in voice_risk: risk_factors.append('认知模糊早期征兆')
    if '站立不稳' in image_risk: risk_factors.append('跌倒风险中')
    if '语调低沉' in voice_risk: risk_factors.append('情绪低落')

    if len(risk_factors) == 0:
        health_status = '正常'
    else:
        # 多风险取最高优先级
        priority = ['认知模糊早期征兆', '跌倒风险中', '心率异常', '血压偏高', '血氧偏低', '情绪低落', '疲劳']
        for p in priority:
            if p in risk_factors:
                health_status = p
                break
        else:
            health_status = '疲劳'

    care_advice = care_advice_templates.get(health_status, "保持健康生活方式，定期体检。")

    record = {
        "user_id": f"elder_{str(random.randint(1, 999999)).zfill(6)}",
        "age": age,
        "gender": gender,
        "chronic_diseases": chronic_diseases,
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "vitals": {
            "heart_rate": heart_rate,
            "blood_pressure": blood_pressure,
            "o2_saturation": o2_saturation,
            "temperature": temperature
        },
        "behavior": {
            "steps_today": steps_today,
            "night_awakenings": night_awakenings,
            "inactivity_duration": inactivity_duration,
            "primary_activity": primary_activity
        },
        "voice_risk": voice_risk,
        "image_risk": image_risk,
        "lab_report": lab_report,
        "teacher_health_status": health_status,
        "teacher_care_advice": care_advice
    }
    return record

# 生成500条数据
dataset = [generate_elder_data() for _ in range(500)]

with open('elder_care_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print("✅ 成功生成 500 条康养模拟数据，已保存为 'elder_care_dataset.json'")