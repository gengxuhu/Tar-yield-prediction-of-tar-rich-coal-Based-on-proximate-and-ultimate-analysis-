import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime
# 读取数据
excel_path = r'D:\研究生学习\博士学习\论文3\数据集\工业分析元素分析焦油产率数据归一化2.xlsx'
train_data = pd.read_excel(excel_path, sheet_name='Atrain')
test_data = pd.read_excel(excel_path, sheet_name='Atest')

# 提取特征和目标变量
X_train = train_data.loc[:, "M":"S"]  # 特征列从M到S
y_train = train_data["Tar"]
X_test = test_data.loc[:, "M":"S"]
y_test = test_data["Tar"]
'''
# 定义TabPFN模型的参数搜索空间
param_dist = {
    'N_ensemble_configurations': [4, 8, 16, 32, 64],  # 集成配置数量
    'feature_shift_decay': np.linspace(0.8, 0.99, 10),  # 特征分布变化的衰减因子
    'base_distribution': ['gaussian', 'laplace'],  # 基础分布类型
    'prior_fit_mode': ['empirical', 'mle'],  # 先验拟合模式
    'batch_size_inference': [256, 512, 1024]  # 推理批大小
}
'''
# 修正后的TabPFN超参数网格
param_grid = {
    'fit_mode': ['prior', 'joint'],  # 有效的拟合模式参数
    'n_estimators': [10, 20],         # 集成模型数量
    'device': ['cpu', 'cuda'],        # 计算设备
    'random_state': [42, 123],        # 随机种子
    'softmax_temperature': [0.5, 1.0] # 软max温度参数
}
# 初始化TabPFN回归器
model = TabPFNRegressor(device='cuda')  # 使用GPU加速

# 设置5折交叉验证
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化随机搜索
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,  # 尝试20种不同的参数组合
    cv=cv,
    scoring='neg_root_mean_squared_error',  # 使用RMSE作为评估指标
    n_jobs=1,  # 并行作业数，设为1避免GPU内存问题
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

# 准备结果存储
results = []

# 打印每个参数组合的训练集和测试集指标
print("\n每个参数组合的评估结果：")
for i, params in enumerate(random_search.cv_results_['params']):
    # 获取当前参数组合的模型
    current_model = TabPFNRegressor(device='cuda', **params)
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

# 创建保存目录（如果不存在）
save_dir = 'D:\研究生学习\博士学习\论文3\数据集\模型参数'
os.makedirs(save_dir, exist_ok=True)

# 生成带时间戳的文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
excel_path = os.path.join(save_dir, f'tabpfn_param_search_{timestamp}.xlsx')

# 保存到Excel
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='参数搜索结果', index=False)
    
    # 添加一个汇总表
    best_result = results_df.iloc[0].copy()
    pd.DataFrame(best_result).T.to_excel(writer, sheet_name='最佳参数', index=False)

print(f"\n所有参数组合的评估结果已保存至: {excel_path}")

