# 导入必要的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from scipy import stats
import matplotlib
import time  # 导入time模块

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

# 读取Excel数据
excel_path = 'D:\研究生学习\博士学习\论文3\数据集\工业分析元素分析焦油产率数据归一化2.xlsx'  # 替换为实际文件路径
train_data = pd.read_excel(excel_path, sheet_name='Btrain')
test_data = pd.read_excel(excel_path, sheet_name='Ctest')

# 提取特征和目标变量
X_train = train_data.loc[:,  "A":"N"]  # 特征列从M到S
y_train = train_data["Tar"]
X_test = test_data.loc[:,  "A":"N"]
y_test = test_data["Tar"]

# 建立随机森林回归模型
#model = RandomForestRegressor(n_estimators=300, random_state=114)
model = RandomForestRegressor(
    n_estimators=50,
    random_state=42, 
    max_depth = 20,
    min_samples_leaf = 4,
    min_samples_split = 2,
    ccp_alpha = 0.1,
    bootstrap = True)

model.fit(X_train, y_train)
# 预测训试集
x_pred = model.predict(X_train)
# 预测测试集
y_pred = model.predict(X_test)
end_time = time.time()


# 计算并输出总运行时间
elapsed_time = end_time - start_time
print(f"\n代码运行时间：{elapsed_time:.2f} 秒")
# 评估模型
MSE = mean_squared_error(y_train, x_pred)
RMSE = np.sqrt(MSE)
MAE = mean_absolute_error(y_train, x_pred)
R2 = r2_score(y_train, x_pred)
train_mape = (abs((y_train - x_pred) / y_train)).mean() * 100

test_mape = (abs((y_test - y_pred) / y_test)).mean() * 100
mse = mean_squared_error(y_test, y_pred)
rmse= np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# 收集评估结果
evaluation_data = {
    '指标': ['训练集MSE', '训练集RMSE', '训练集MAE', '训练集R²', '训练集MAPE',
            '测试集MSE', '测试集RMSE', '测试集MAE', '测试集R²', '测试集MAPE'],
    '值': [MSE, RMSE, MAE, R2, train_mape,
          mse, rmse, mae, r2, test_mape]
}

# 将评估结果转换为DataFrame
evaluation_df = pd.DataFrame(evaluation_data)
# 打印评估结果
print(f'Mean Squared Error (MSE): {MSE:.4f}')
print(f'Root Mean Squared Error (RMSE): {RMSE:.4f}')
print(f'Mean Absolute Error (MAE): {MAE:.4f}')
print(f'R² Score: {R2:.4f}')
print(f'Mean Absolute Percentage Error (MAPE) for Training Data: {train_mape:.4f}%')
print(f'Mean Absolute Percentage Error (MAPE) for Testing Data: {test_mape:.4f}%')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'R² Score: {r2:.4f}')


feature_importance = model.feature_importances_
features = X_train.columns
# 将特征名和重要性值转换为DataFrame
feature_importances_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})

# 排序特征重要性（可选）
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

# 将DataFrame保存到Excel文件
feature_importances_df.to_excel(r'D:\研究生学习\博士学习\论文3\数据集\模型参数\RF特征重要性B.xlsx')


# 创建包含真实Y值和预测Y值的DataFrame
train_results = pd.DataFrame({'真实Y值':y_train.values , '预测Y值': x_pred})
test_results = pd.DataFrame({'真实Y值': y_test.values, '预测Y值': y_pred})

# 将DataFrame写入Excel文件
with pd.ExcelWriter('D:\研究生学习\博士学习\论文3\数据集\模型参数\RF训练测试结果盆地B.xlsx') as writer:
    train_results.to_excel(writer, sheet_name='train', index=False)
    test_results.to_excel(writer, sheet_name='test', index=False)
    evaluation_df.to_excel(writer, sheet_name='评估结果', index=False)

