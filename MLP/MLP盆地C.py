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
# 导入MLP相关库
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
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
train_data = pd.read_excel(excel_path, sheet_name='Ctrain')
test_data = pd.read_excel(excel_path, sheet_name='Btest')

# 提取特征和目标变量
X_train = train_data.loc[:,  "A":"N"]  # 特征列从M到S
y_train = train_data["Tar"]
X_test = test_data.loc[:,  "A":"N"]
y_test = test_data["Tar"]




# 建立MLP回归模型
model = MLPRegressor(
    hidden_layer_sizes=(100, 100),  # 增加隐藏层数量和神经元数量，增强模型复杂度
    activation='logistic',  # 激活函数，'relu' 表示使用修正线性单元函数，用于在神经元中引入非线性特性
    solver='lbfgs',  # 权重优化的求解器，'adam' 是一种基于自适应矩估计的优化算法，适用于较大数据集和较高维度的问题
    alpha=0.1,  # 减小L2正则化项参数，降低正则化强度，帮助模型更好拟合数据
    batch_size='auto',  # 训练模型时的批次大小，'auto' 表示批次大小为min(200, n_samples)
    learning_rate='invscaling',  # 学习率策略，'adaptive' 表示当训练损失在连续两次迭代中没有改善时，学习率会自动降低
    max_iter=300,  # 增加最大迭代次数，让模型有更多机会学习数据特征
    #early_stopping=True,  # 是否启用早停机制，若为True，则在验证集上的性能连续多次迭代没有改善时停止训练
    #validation_fraction=0.1,  # 当启用早停机制时，从训练集中划分出作为验证集的比例，这里表示划分10%的数据作为验证集
    random_state=42  # 随机数种子，保证每次运行代码时模型的初始化和数据划分结果一致，方便复现实验结果
)
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

# 创建包含真实Y值和预测Y值的DataFrame
train_results = pd.DataFrame({'真实Y值':y_train.values , '预测Y值': x_pred})
test_results = pd.DataFrame({'真实Y值': y_test.values, '预测Y值': y_pred})


# feature_importances_df.to_excel(...)

# 修改结果文件名
with pd.ExcelWriter('D:\研究生学习\博士学习\论文3\数据集\模型参数\MLP训练测试结果盆地C.xlsx') as writer:
    train_results.to_excel(writer, sheet_name='train', index=False)
    test_results.to_excel(writer, sheet_name='test', index=False)
    evaluation_df.to_excel(writer, sheet_name='评估结果', index=False)

