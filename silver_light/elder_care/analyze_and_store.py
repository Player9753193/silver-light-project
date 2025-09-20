# analyze_and_store.py
import json
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import os
import csv
import matplotlib

# =================== 设置中文字体 ===================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
matplotlib.use('Agg')

# =================== 配置 ===================
INPUT_FILE = "results.json"
DB_FILE = "elder_care.db"
PLOT_DIR = "plots"
CSV_FILE = "elder_care.csv"
os.makedirs(PLOT_DIR, exist_ok=True)


# =================== 数据库初始化 ===================
def init_database():
    """创建 SQLite 数据库和表"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            user_id TEXT PRIMARY KEY,
            normal REAL,
            fatigue REAL,
            heart_abnormal REAL,
            high_bp REAL,
            low_o2 REAL,
            low_mood REAL,
            fall_risk REAL,
            cognitive_risk REAL,
            primary_risk TEXT,
            risk_level TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()


# =================== 高风险筛选规则 ===================
def classify_risk(user_id: str, probs: Dict[str, float]) -> Tuple[str, str]:
    states = list(probs.keys())
    values = list(probs.values())
    primary_risk = states[np.argmax(values)]
    max_prob = max(values)

    multi_risk_count = len([v for v in probs.values() if v > 0.12]) >= 3

    if max_prob > 0.20 or multi_risk_count:
        risk_level = "高风险"
    elif max_prob > 0.15 or len([v for v in probs.values() if v > 0.15]) >= 2:
        risk_level = "中风险"
    else:
        risk_level = "低风险"

    return primary_risk, risk_level


# =================== 存入数据库 ===================
def save_to_db(data: List[Dict]):
    """将分析结果存入数据库"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for item in data:
        cursor.execute('''
            INSERT OR REPLACE INTO predictions 
            (user_id, normal, fatigue, heart_abnormal, high_bp, low_o2, 
             low_mood, fall_risk, cognitive_risk, primary_risk, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item['user_id'],
            item['probs']['正常'],
            item['probs']['疲劳'],
            item['probs']['心率异常'],
            item['probs']['血压偏高'],
            item['probs']['血氧偏低'],
            item['probs']['情绪低落'],
            item['probs']['跌倒风险中'],
            item['probs']['认知模糊早期征兆'],
            item['primary_risk'],
            item['risk_level']
        ))

    conn.commit()
    conn.close()
    print(f"✅ 数据已保存到数据库: {DB_FILE}")


# =================== 导出 CSV 文件===================
def export_csv():
    """从数据库导出 CSV 文件，带 UTF-8 with BOM，确保 Excel 正确识别"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions")
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]

    # utf-8编码，自动添加BOM，Excel可正确识别中文
    with open(CSV_FILE, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        writer.writerows(rows)

    conn.close()
    print(f"✅ 数据已导出为 CSV 文件（Excel 友好）: {CSV_FILE}")


# =================== 可视化：柱状图 ===================
def plot_group_bar(all_data: List[Dict]):
    """绘制各类风险的平均概率"""
    labels = ["正常", "疲劳", "心率异常", "血压偏高", "血氧偏低", "情绪低落", "跌倒风险中", "认知模糊早期征兆"]
    avg_probs = [np.mean([d['probs'][label] for d in all_data]) for label in labels]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, avg_probs, color='skyblue', edgecolor='navy', alpha=0.8)
    plt.title("群体健康状态平均概率分布")
    plt.ylabel("概率")
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/group_risk_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 群体分布图已生成")


# =================== 可视化：个人雷达图 ===================
def plot_radar(user_id: str, probs: Dict[str, float]):
    """绘制单个用户的雷达图"""
    labels = ["正常", "疲劳", "心率异常", "血压偏高", "血氧偏低", "情绪低落", "跌倒风险中", "认知模糊早期征兆"]
    values = [probs[label] for label in labels]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='red', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    plt.title(f"健康风险雷达图: {user_id}", pad=20, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{user_id}_radar.png", dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            raw_results = json.load(f)
    except Exception as e:
        print(f"❌ 读取 {INPUT_FILE} 失败: {e}")
        return

    analyzed_data = []
    high_risk_list = []
    mid_risk_list = []

    for user_id, probs in raw_results.items():
        if "error" in probs:
            continue

        primary_risk, risk_level = classify_risk(user_id, probs)
        item = {
            "user_id": user_id,
            "probs": probs,
            "primary_risk": primary_risk,
            "risk_level": risk_level
        }
        analyzed_data.append(item)

        if risk_level == "高风险":
            high_risk_list.append(item)
        elif risk_level == "中风险":
            mid_risk_list.append(item)

        plot_radar(user_id, probs)

    plot_group_bar(analyzed_data)
    init_database()
    save_to_db(analyzed_data)
    export_csv()

    print("\n" + "="*50)
    print("              🚨 高风险用户报告")
    print("="*50)
    print(f"📊 总人数: {len(analyzed_data)}")
    print(f"🔴 高风险: {len(high_risk_list)} 人")
    print(f"🟡 中风险: {len(mid_risk_list)} 人")
    print(f"🟢 低风险: {len(analyzed_data) - len(high_risk_list) - len(mid_risk_list)} 人")
    print("\n🔴 高风险用户:")
    for item in high_risk_list:
        print(f"   {item['user_id']} → 主要风险: {item['primary_risk']}")

    with open("high_risk_elders.json", "w", encoding="utf-8") as f:
        json.dump(high_risk_list, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 高风险名单已保存: high_risk_elders.json")
    print(f"✅ 所有雷达图已保存至: {PLOT_DIR}/")
    print(f"✅ 数据库已更新: {DB_FILE}")
    print(f"✅ 数据已导出为 CSV 文件（Excel 可读）: {CSV_FILE}")


if __name__ == "__main__":
    main()