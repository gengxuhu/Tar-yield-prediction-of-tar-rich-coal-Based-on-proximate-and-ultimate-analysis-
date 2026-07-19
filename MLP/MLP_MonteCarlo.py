# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import os

# 配置matplotlib字体
config = {
    "font.family": 'serif',
    "font.size": 12,
    "mathtext.fontset": 'stix',
    "font.serif": ['STSong'],  # 宋体
    'axes.unicode_minus': False
}
rcParams.update(config)

def run_monte_carlo_simulation():
    # 1. 数据加载
    excel_path = r'D:\研究生学习\博士学习\论文3\数据集\工业分析元素分析焦油产率数据归一化2.xlsx'
    print(f"正在加载数据: {excel_path}")
    
    try:
        train_data = pd.read_excel(excel_path, sheet_name='alltrain')
        test_data = pd.read_excel(excel_path, sheet_name='alltest')
    except FileNotFoundError:
        print(f"错误: 找不到文件 {excel_path}")
        return
    except Exception as e:
        print(f"读取数据时发生错误: {e}")
        return

    # 提取特征和目标变量
    X_train = train_data.loc[:, "A":"N"]
    y_train = train_data["Tar"]
    X_test = test_data.loc[:, "A":"N"]
    y_test = test_data["Tar"]

    # 2. 蒙特卡洛模拟参数设置
    n_iterations = 100
    results = []
    
    print(f"开始执行蒙特卡洛模拟，共 {n_iterations} 次迭代...")
    start_total_time = time.time()

    # 3. 循环迭代
    for i in range(n_iterations):
        # 生成随机种子
        seed = np.random.randint(0, 100000)
        
        # 建立MLP回归模型 (使用当前随机种子)
        model = MLPRegressor(
            hidden_layer_sizes=(50, 50),
            activation='tanh',
            solver='lbfgs',
            alpha=0.1,
            batch_size='auto',
            learning_rate='constant',
            max_iter=400,
            random_state=seed  # 关键：每次迭代使用不同的随机种子
        )
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # 计算指标
        # 训练集
        train_mse = mean_squared_error(y_train, train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        train_mape = (abs((y_train - train_pred) / y_train)).mean() * 100
        
        # 测试集
        test_mse = mean_squared_error(y_test, test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_mape = (abs((y_test - test_pred) / y_test)).mean() * 100
        
        # 记录结果
        results.append({
            'Iteration': i + 1,
            'Seed': seed,
            'Train_MSE': train_mse,
            'Train_RMSE': train_rmse,
            'Train_MAE': train_mae,
            'Train_R2': train_r2,
            'Train_MAPE': train_mape,
            'Test_MSE': test_mse,
            'Test_RMSE': test_rmse,
            'Test_MAE': test_mae,
            'Test_R2': test_r2,
            'Test_MAPE': test_mape
        })
        
        if (i + 1) % 10 == 0:
            print(f"已完成 {i + 1}/{n_iterations} 次迭代")

    total_time = time.time() - start_total_time
    print(f"模拟完成，总耗时: {total_time:.2f} 秒")

    # 4. 结果整理与统计
    results_df = pd.DataFrame(results)
    
    # 计算统计量
    stats_df = results_df.describe().loc[['mean', 'std', 'min', 'max']]
    
    # 5. 可视化
    # 创建保存目录
    output_dir = os.path.dirname(excel_path)
    if not os.path.exists(output_dir):
        output_dir = os.getcwd() # 如果原路径不存在，保存在当前目录
        
    # R2 分布图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(results_df['Test_R2'], bins=20, color='skyblue', edgecolor='black')
    plt.title('测试集 R² 分布直方图')
    plt.xlabel('R²')
    plt.ylabel('频数')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(results_df['Test_R2'], vert=True, patch_artist=True)
    plt.title('测试集 R² 箱线图')
    plt.ylabel('R²')
    
    r2_plot_path = os.path.join(output_dir, 'MonteCarlo_R2_Distribution.png')
    plt.tight_layout()
    plt.savefig(r2_plot_path)
    print(f"R2分布图已保存: {r2_plot_path}")
    
    # RMSE 分布图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(results_df['Test_RMSE'], bins=20, color='salmon', edgecolor='black')
    plt.title('测试集 RMSE 分布直方图')
    plt.xlabel('RMSE')
    plt.ylabel('频数')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(results_df['Test_RMSE'], vert=True, patch_artist=True)
    plt.title('测试集 RMSE 箱线图')
    plt.ylabel('RMSE')
    
    rmse_plot_path = os.path.join(output_dir, 'MonteCarlo_RMSE_Distribution.png')
    plt.tight_layout()
    plt.savefig(rmse_plot_path)
    print(f"RMSE分布图已保存: {rmse_plot_path}")

    # 6. 保存Excel结果
    output_excel_path = os.path.join(output_dir, 'MLP_MonteCarlo_Results.xlsx')
    try:
        with pd.ExcelWriter(output_excel_path) as writer:
            results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            stats_df.to_excel(writer, sheet_name='Statistics')
        print(f"详细结果已保存至 Excel: {output_excel_path}")
    except Exception as e:
        print(f"保存Excel失败: {e}")

    # 打印简要报告
    print("\n" + "="*30)
    print("蒙特卡洛模拟统计报告 (测试集)")
    print("="*30)
    print(f"平均 R²: {results_df['Test_R2'].mean():.4f} ± {results_df['Test_R2'].std():.4f}")
    print(f"最大 R²: {results_df['Test_R2'].max():.4f}")
    print(f"最小 R²: {results_df['Test_R2'].min():.4f}")
    print("-" * 30)
    print(f"平均 RMSE: {results_df['Test_RMSE'].mean():.4f} ± {results_df['Test_RMSE'].std():.4f}")
    print(f"最小 RMSE: {results_df['Test_RMSE'].min():.4f}")
    print(f"最大 RMSE: {results_df['Test_RMSE'].max():.4f}")
    print("="*30)

if __name__ == "__main__":
    run_monte_carlo_simulation()
