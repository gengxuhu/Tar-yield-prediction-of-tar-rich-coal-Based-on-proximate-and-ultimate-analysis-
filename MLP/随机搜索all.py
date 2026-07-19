# 导入必要的库
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from scipy import stats
import matplotlib
import time  # 导入time模块
import os
from sklearn.neural_network import MLPRegressor
# 导入TabPFN模型
from tabpfn import TabPFNRegressor

matplotlib.use('TkAgg')

# 记录程序开始时间
start_time = time.time()

config = {
    "font.family": 'serif',
    "font.size": 12,  # 相当于小四大小
    "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ['STSong'],  # 宋体
    'axes.unicode_minus': False  # 处理负号，即-号
}
rcParams.update(config)

# 修改数据读取部分：读取已划分好的训练集和测试集
# 请确保Excel文件路径正确
excel_path = r'D:\研究生学习\博士学习\论文3\数据集\工业分析元素分析焦油产率数据归一化2.xlsx'  # 替换为实际文件路径
train_data = pd.read_excel(excel_path, sheet_name='alltrain')
test_data = pd.read_excel(excel_path, sheet_name='alltest')

# 提取特征和目标变量
X_train = train_data.loc[:,  "A":"N"]  # 特征列从M到S
y_train = train_data["Tar"]
X_test = test_data.loc[:,  "A":"N"]
y_test = test_data["Tar"]

# 定义MLP模型超参数分布，为模型进行参数寻优
param_dist = {
    'hidden_layer_sizes': [ (50, 50), (100, 50), (100, 100)],  # 隐藏层结构
    'activation': ['relu', 'tanh', 'logistic'],  # 激活函数
    'solver': ['adam', 'lbfgs', 'sgd'],  # 优化算法
    'solver': ['adam', 'lbfgs', 'sgd'],  # 优化算法
    'alpha': [0.0001, 0.001, 0.01, 0.1],  # L2正则化参数
    'learning_rate': ['constant', 'invscaling', 'adaptive'],  # 学习率策略
    'max_iter': [100, 200, 300, 400],  # 最大迭代次数
    'batch_size': [16, 32, 64, 128]  # 批量大小
}
# 创建MLP模型
model = MLPRegressor(random_state=42)

# 设置5折交叉验证
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化随机搜索
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=100,  # 尝试30种不同的参数组合
    cv=cv,
    scoring='neg_root_mean_squared_error',  # 使用RMSE作为评估指标
    n_jobs=-1,  # 使用所有CPU核心
    verbose=2,  # 显示详细信息
    random_state=42
)

# 执行随机搜索
print("开始随机搜索最优参数...")
random_search.fit(X_train, y_train)

# 获取最优模型
best_model = random_search.best_estimator_
print(f"最佳参数组合: {random_search.best_params_}")
print(f"交叉验证最佳RMSE: {-random_search.best_score_:.4f}")

# 在测试集上评估性能
y_test_pred = best_model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)

print("\n测试集最终评估指标:")
print(f"R²: {test_r2:.4f}, MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}")

# 准备结果存储
results = []

# 打印每个参数组合的训练集和测试集指标
print("\n每个参数组合的评估结果：")
for i, params in enumerate(random_search.cv_results_['params']):
    # 获取当前参数组合的模型
    current_model = MLPRegressor(random_state=42, **params)
    current_model.fit(X_train, y_train)
    
    # 在训练集和测试集上预测
    y_train_pred = current_model.predict(X_train)
    y_test_pred = current_model.predict(X_test)
    
    # 计算评估指标
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    
    # 保存结果
    result = {
        '参数组合': i + 1,
        **params,
        '训练集_R2': train_r2,
        '训练集_MAE': train_mae,
        '训练集_MSE': train_mse,
        '训练集_RMSE': train_rmse,
        '测试集_R2': test_r2,
        '测试集_MAE': test_mae,
        '测试集_MSE': test_mse,
        '测试集_RMSE': test_rmse,
        '交叉验证_RMSE': -random_search.cv_results_['mean_test_score'][i]
    }
    results.append(result)
    
    # 打印结果
    print(f"\n参数组合 {i + 1}:")
    print(f"参数: {params}")
    print(f"训练集 - R²: {train_r2:.4f}, MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}")
    print(f"测试集 - R²: {test_r2:.4f}, MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}")
    print(f"交叉验证 - RMSE: {-random_search.cv_results_['mean_test_score'][i]:.4f}")

# 转换为DataFrame并保存到Excel
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='测试集_R2', ascending=False)  # 按测试集R²排序

# 指定保存文件路径
save_dir = r'D:\研究生学习\博士学习\论文3\数据集\模型参数'
excel_path = os.path.join(save_dir, 'MLP随机搜索结果all.xlsx')

# 保存到Excel
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='参数搜索结果', index=False)
    
    # 添加一个汇总表
    best_result = results_df.iloc[0].copy()
    pd.DataFrame(best_result).T.to_excel(writer, sheet_name='最佳参数', index=False)

print(f"\n所有参数组合的评估结果已保存至: {excel_path}")

# 计算并输出总运行时间
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n代码运行时间：{elapsed_time:.2f} 秒")