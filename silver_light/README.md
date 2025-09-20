
README.md

这是银光计划(silver light project)的核心代码，实现了一个基于知识蒸馏的轻量级健康风险评估模型。该模型接收老年人的11维生理与行为结构化指标输入，输出8类健康状态的软标签预测，用于评估跌倒、心率、血压、认知等风险。此模型可作为康养系统中进行快速风险筛查和个性化建议生成的基础模块。
MLPWithAttention 模型是一个为知识蒸馏设计的轻量级神经网络。其结构为：MLP特征提取层 + 特殊形式的Self-Attention层 + MLP分类头。目标是接收11维的结构化特征输入，并输出8类健康状态的预测（软标签）。该模型通过MLP提取基础特征，再利用一个作用于单点序列的Self-Attention层（模拟特征间交互或进行特征变换），最后通过分类头输出结果。

# 0. 安装依赖
```bash
pip install -r requirements.txt
```

1. 生成初始数据集
```bash
python generate_init_dataset.py
```

2. 准备教师模型输入数据（含模拟标签）
```bash
python prepare_for_teacher_with_pseudo.py
```

3. 调用教师模型（如qwen-max）进行软标签标注
```bash
python teacher_labeling_batch.py
```

4. 转换教师模型输出为学生模型输入格式
```bash
python convert_teacher_output.py --input ../elder_care_teachered_softlabels.json --output ./student_input.json
```

5. 数据预处理与分割
```bash
python data_preprocess.py --input student_input.json --output_dir ./student_data --normalize standard
```

6. 训练学生模型
```bash
python train_curve.py --data_dir ./student_data --epochs 100 --lr 0.001 --save_model ./student_data/student_mlp_attn_best.pth
```

7. 准备推理用数据集
```bash
python prepare_inference_data.py --input ../elder_care_dataset.json --output inference_input.json
```

8. 模型推理
```bash
python inference.py --input inference_input.json --output results.json
```

9. 结果分析与可视化
```bash
python analyze_and_store.py
```

10. 导出 ONNX 模型（用于部署）
```bash
python export_onnx.py
```


#############################################################################

***1. 特征和软标签***
✅ 构建输入特征矩阵: (14, 11)
✅ 构建软标签矩阵: (14, 8)

***2. 模型训练参数***
✅ 归一化参数: ./student_data/scaler_info.json
✅ 特征矩阵: ./student_data/X_features.npy
✅ 软标签矩阵: ./student_data/Y_soft_labels.npy

***3. 模型的训练过程***
Epoch [ 10/100] | Train Loss: 0.1691 | Val Loss: 0.0693 | Patience: 2/15
✅ 模型保存: ./student_data/student_mlp_attn_best.pth (Val Loss: 0.0656)
Epoch [ 20/100] | Train Loss: 0.0742 | Val Loss: 0.0660 | Patience: 1/15
Epoch [ 30/100] | Train Loss: 0.0528 | Val Loss: 0.0777 | Patience: 11/15
📢 早停触发！最佳验证损失: 0.0656
📈 损失曲线已保存: loss_curve.png
🎉 训练完成！最终模型保存为: ./student_data/student_mlp_attn_best.pth


***4. 用户分析，可视化群体柱状图***
高风险名单示例并保存: high_risk_elders.json

==================================================
              🚨 高风险用户报告
==================================================
📊 总人数: 500
🔴 高风险: 499 人
🟡 中风险: 1 人
🟢 低风险: 0 人

🔴 高风险用户:
   elder_198554 → 主要风险: 心率异常
   elder_654228 → 主要风险: 血压偏高
   elder_377393 → 主要风险: 跌倒风险中
   elder_354821 → 主要风险: 跌倒风险中
   elder_256429 → 主要风险: 心率异常
   elder_676407 → 主要风险: 正常
   elder_320864 → 主要风险: 正常
   elder_521712 → 主要风险: 血压偏高
   elder_707902 → 主要风险: 血压偏高
   elder_012469 → 主要风险: 心率异常
   elder_471830 → 主要风险: 血压偏高
   elder_998913 → 主要风险: 血压偏高
   elder_195761 → 主要风险: 认知模糊早期征兆
   elder_407889 → 主要风险: 血压偏高
   elder_258831 → 主要风险: 血压偏高
   elder_768863 → 主要风险: 正常
   elder_080252 → 主要风险: 认知模糊早期征兆
   elder_743545 → 主要风险: 血压偏高
   elder_409836 → 主要风险: 血压偏高
   elder_395728 → 主要风险: 认知模糊早期征兆
   elder_537396 → 主要风险: 血压偏高
   elder_854679 → 主要风险: 血压偏高
   elder_698741 → 主要风险: 心率异常
   elder_522953 → 主要风险: 血压偏高
   elder_359377 → 主要风险: 血压偏高
   ......


#############################################################################

project/
├── generate_init_dataset.py    					# 生成数据集
├── prepare_for_teacher_with_pseudo.py      		
├── elder_care_for_teacher_with_pseudo.json 		# 带模拟标签的教师数据集         				
├── teacher_labeling_batch.pth  					# 教师模型打标
├── elder_care_dataset.json    						# 原始数据
└── elder_care/
    ├── student_model.py							# 模型定义
	├── prepare_inference_data.py
    ├── train_curve.py          					# 模型训练可视化
    ├── inference.py								# 推理脚本
    ├── export_onnx.py
    ├── requirements.txt
    ├── student_mlp_attn_best.pth   				# 训练后生成模型权重
    ├── loss_curve.png              				# 训练可视化曲线图
    ├── elder_care_teachered_softlabels.json        # 原始输入，教师打标结果
    ├── convert_teacher_output.py  					# 转换脚本
    ├── student_input.json         					# 每行一个 JSON
    ├── data_preprocess.py         					# 数据预处理脚本
    └── student_data/
        ├── X_features.npy
        ├── Y_soft_labels.npy
        ├── scaler_info.json						# 归一化参数
        └── feature_names.json
    ├── results.json                  				# 推理结果
    ├── high_risk_elders.json         				# 高风险用户名单
    ├── elder_care.db                 				# 输出的SQLite 数据库
	├── elder_care.csv                				# CSV 输出文件
    ├── plots/
    │   ├── group_risk_distribution.png     		# 群体柱状图
    │   ├── elder_391247_radar.png          		# 个人雷达图
    │   ├── elder_916221_radar.png
    │   └── ... 
    └── analyze_and_store.py