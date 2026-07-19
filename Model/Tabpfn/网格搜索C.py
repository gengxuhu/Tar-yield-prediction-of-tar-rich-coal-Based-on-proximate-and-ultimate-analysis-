import pandas as pd
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from itertools import product
import time

# 记录程序开始时间
start_time = time.time()

# 数据读取
excel_path = 'D:\\研究生学习\\博士学习\\论文3\\数据集\\工业分析元素分析焦油产率数据归一化2.xlsx'
train_data = pd.read_excel(excel_path, sheet_name='Ctrain')
test_data = pd.read_excel(excel_path, sheet_name='Ctest')

# 特征及目标
X_train = train_data.loc[:, "M":"S"]
y_train = train_data["Tar"]
X_test = test_data.loc[:, "M":"S"]
y_test = test_data["Tar"]

# 设置TabPFN核心参数网格
param_grid = {
    'n_estimators': [8, 16],  # TabPFN推荐较小数量
    'fit_mode': ['regression'],  # 回归问题
    'device': ['cpu'],  # 可改为'cuda'加速
    'softmax_temperature': [1.0, 2.0],  # 平滑度
    'random_state': [42]  # 保持一致
}

results = []
param_names = list(param_grid.keys())
param_combinations = list(product(*param_grid.values()))

for i, params in enumerate(param_combinations, 1):
    param_dict = dict(zip(param_names, params))
    print(f"\n=== 迭代 {i}/{len(param_combinations)} ===")
    print(f"超参数配置: {param_dict}")

    # 初始化TabPFN回归模型
    model = TabPFNRegressor(**param_dict)

    # 训练模型
    model.fit(X_train.values, y_train.values)

    # 预测
    y_train_pred = model.predict(X_train.values)
    y_test_pred = model.predict(X_test.values)

    # 评估指标
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)

    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)

    print("训练集指标:")
    print(f"  R²: {train_r2:.4f}, MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}")
    print("测试集指标:")
    print(f"  R²: {test_r2:.4f}, MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}")

    results.append({
        '迭代次数': i,
        **param_dict,
        '训练集R²': train_r2,
        '训练集MAE': train_mae,
        '训练集MSE': train_mse,
        '训练集RMSE': train_rmse,
        '测试集R²': test_r2,
        '测试集MAE': test_mae,
        '测试集MSE': test_mse,
        '测试集RMSE': test_rmse
    })

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n代码运行时间：{elapsed_time:.2f} 秒")

# 保存结果
results_df = pd.DataFrame(results)
save_path = r'D:\研究生学习\博士学习\论文3\数据集\模型参数\TabPFN网格搜索结果C.xlsx'
with pd.ExcelWriter(save_path) as writer:
    results_df.to_excel(writer, sheet_name='参数迭代结果', index=False)

print(f"TabPFN网格搜索完成，结果已保存到'{save_path}'")