"""
测试SHAP逆变换功能
验证SHAPRF-ALL.py中的逆变换功能是否正确工作
"""

import pandas as pd
import numpy as np
import pickle
import os

# 检查minmax_scaler.pkl文件是否存在
scaler_path = os.path.join(os.path.dirname(__file__), 'minmax_scaler.pkl')
print(f"检查归一化参数文件: {scaler_path}")

if os.path.exists(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"成功加载归一化参数文件")
    print(f"数据范围 (data_min_): {scaler.data_min_}")
    print(f"数据范围 (data_max_): {scaler.data_max_}")
    print(f"特征数量: {len(scaler.data_min_)}")
else:
    print(f"错误: 未找到归一化参数文件 {scaler_path}")

# 读取原始数据用于验证
print("\n读取原始数据用于验证...")
excel_path = r'D:\研究生学习\博士学习\论文3\数据集\工业分析元素分析焦油产率数据归一化2.xlsx'
train_data = pd.read_excel(excel_path, sheet_name='alltrain')
print(f"可用列: {train_data.columns.tolist()}")
# 使用实际可用的特征
X_train = train_data.loc[:, "A":"N"]

print(f"原始数据形状: {X_train.shape}")
print(f"原始数据前5行:")
print(X_train.head())

# 如果找到scaler，测试逆变换
if 'scaler' in locals() and scaler is not None:
    print("\n测试逆变换功能...")
    # 对原始数据进行归一化
    X_train_normalized = scaler.transform(X_train)
    print(f"归一化数据形状: {X_train_normalized.shape}")
    print(f"归一化数据前5行:")
    print(X_train_normalized[:5])
    
    # 进行逆变换
    X_train_restored = scaler.inverse_transform(X_train_normalized)
    print(f"逆变换后数据形状: {X_train_restored.shape}")
    print(f"逆变换后数据前5行:")
    print(X_train_restored[:5])
    
    # 检查逆变换是否成功
    diff = np.abs(X_train.values - X_train_restored)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\n逆变换精度检查:")
    print(f"最大差异: {max_diff}")
    print(f"平均差异: {mean_diff}")
    
    if max_diff < 1e-10:
        print("✓ 逆变换成功，数据完全恢复")
    else:
        print("⚠ 逆变换存在微小差异")
else:
    print("\n无法测试逆变换功能，因为未找到归一化参数")