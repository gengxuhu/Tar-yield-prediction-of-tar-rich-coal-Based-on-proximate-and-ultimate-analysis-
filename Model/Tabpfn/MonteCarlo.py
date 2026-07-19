# 导入必要的库
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from scipy import stats
import matplotlib
import time
import seaborn as sns
from tqdm import tqdm
from tabpfn import TabPFNRegressor

matplotlib.use('TkAgg')

# 配置绘图参数
config = {
    "font.family": 'serif',
    "font.size": 12,
    "mathtext.fontset": 'stix',
    "font.serif": ['STSong'],
    'axes.unicode_minus': False
}
rcParams.update(config)

# ==========================================
# 1. 数据准备
# ==========================================
excel_path = 'D:\研究生学习\博士学习\论文3\数据集\工业分析元素分析焦油产率数据归一化2.xlsx'
try:
    train_data = pd.read_excel(excel_path, sheet_name='alltrain')
    test_data = pd.read_excel(excel_path, sheet_name='alltest')
except FileNotFoundError:
    print(f"错误：找不到文件 {excel_path}")
    # 仅用于演示，实际运行需确保文件存在
    pass

# 提取特征和目标变量
X_train_raw = train_data.loc[:, "A":"N"]
y_train = train_data["Tar"]
X_test = test_data.loc[:, "A":"N"]
y_test = test_data["Tar"]

print(f"原始训练集大小: {X_train_raw.shape}")
print(f"测试集(验证集)大小: {X_test.shape}")

# ==========================================
# 2. 蒙特卡洛模拟设置
# ==========================================
N_SIMULATIONS = 100
NOISE_MU = 0
NOISE_SIGMA = 0.1

# 存储结果的容器
simulation_results = {
    'mse': [], 'rmse': [], 'mae': [], 'r2': [],
    'predictions': [],  # 存储每次对测试集的预测结果
    'feature_importances': [] # TabPFN需要通过排列重要性计算
}

# 定义排列重要性计算函数 (因为TabPFN没有内置feature_importances_)
def calculate_permutation_importance(model, X_val, y_val, metric=r2_score):
    """计算特征重要性：通过打乱某一列特征，观察模型性能下降程度"""
    X_val_np = X_val if isinstance(X_val, np.ndarray) else X_val.values.astype(np.float32)
    y_val_np = y_val if isinstance(y_val, np.ndarray) else y_val.values.astype(np.float32)
    
    baseline_pred = model.predict(X_val_np)
    baseline_score = metric(y_val_np, baseline_pred)
    
    importances = []
    for col in range(X_val_np.shape[1]):
        save_col = X_val_np[:, col].copy()
        # 打乱该列
        np.random.shuffle(X_val_np[:, col])
        
        # 重新预测并评分
        perm_pred = model.predict(X_val_np)
        perm_score = metric(y_val_np, perm_pred)
        
        # 恢复该列
        X_val_np[:, col] = save_col
        
        # 重要性 = 基准分数 - 打乱后分数
        importances.append(baseline_score - perm_score)
        
    return np.array(importances)

print(f"\n开始执行蒙特卡洛模拟 (共 {N_SIMULATIONS} 次)...")
start_time = time.time()

# ==========================================
# 3. 循环执行模拟
# ==========================================
# 初始化模型 (TabPFN可以在循环外初始化，fit时只是加载数据)
# 注意：TabPFNRegressor的fit非常快，因为它不进行传统意义上的训练，而是构建上下文
model = TabPFNRegressor(device='cuda') 

for i in tqdm(range(N_SIMULATIONS), desc="Simulating"):
    # 3.1 添加随机噪声到训练集特征
    # 保持验证集 X_test 不变
    noise = np.random.normal(NOISE_MU, NOISE_SIGMA, X_train_raw.shape)
    X_train_noisy = X_train_raw + noise
    
    # 转换为float32 (TabPFN要求)
    X_train_np = X_train_noisy.astype(np.float32)
    y_train_np = y_train.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)
    
    # 3.2 训练 (Fit)
    model.fit(X_train_np, y_train_np)
    
    # 3.3 预测
    y_pred = model.predict(X_test_np)
    
    # 3.4 计算指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 3.5 计算特征重要性 (使用测试集计算排列重要性)
    # 注意：这会增加计算时间，但能提供必要的稳定性分析
    feat_imp = calculate_permutation_importance(model, X_test_np, y_test, r2_score)
    
    # 3.6 存储结果
    simulation_results['mse'].append(mse)
    simulation_results['rmse'].append(rmse)
    simulation_results['mae'].append(mae)
    simulation_results['r2'].append(r2)
    simulation_results['predictions'].append(y_pred)
    simulation_results['feature_importances'].append(feat_imp)

elapsed_time = time.time() - start_time
print(f"\n模拟完成，总耗时：{elapsed_time:.4f} 秒")

# ==========================================
# 4. 数据统计与稳定性分析
# ==========================================

r2_scores = np.array(simulation_results['r2'])
mse_scores = np.array(simulation_results['mse'])
predictions_matrix = np.array(simulation_results['predictions']).T
feature_importances_matrix = np.array(simulation_results['feature_importances'])

# 4.1 基础统计
mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)
cv_r2 = (std_r2 / mean_r2) * 100 if mean_r2 != 0 else 0

mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)

# 95% 置信区间
ci_r2 = stats.t.interval(0.95, len(r2_scores)-1, loc=mean_r2, scale=stats.sem(r2_scores))

print("\n" + "="*30)
print("TabPFN 蒙特卡洛模拟评估报告")
print("="*30)
print(f"R² 均值: {mean_r2:.4f}")
print(f"R² 标准差: {std_r2:.4f}")
print(f"R² 变异系数 (CV): {cv_r2:.2f}%")
print(f"R² 95% 置信区间: ({ci_r2[0]:.4f}, {ci_r2[1]:.4f})")
print(f"MSE 均值: {mean_mse:.4f}")

# 4.2 稳定性判定
is_stable = cv_r2 < 5
print("-" * 30)
print(f"稳定性判定: {'[稳定]' if is_stable else '[不稳定]'} (阈值: CV < 5%)")

# 4.3 异常值检测 (3-sigma 原则)
z_scores = np.abs(stats.zscore(r2_scores))
outliers = np.where(z_scores > 3)[0]
outlier_ratio = len(outliers) / N_SIMULATIONS * 100
print(f"异常值比例: {outlier_ratio:.1f}% (阈值 < 10%)")
if outlier_ratio > 10:
    print("警告：随机噪声导致了过多的异常结果！")

# ==========================================
# 5. 可视化结果
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# 图1：R² 和 MSE 的箱线图
sns.boxplot(data=pd.DataFrame({'R²': r2_scores, 'MSE': mse_scores}), ax=axes[0, 0])
axes[0, 0].set_title('性能指标分布 (箱线图)')
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

# 图2：R² 的时序波动图
axes[0, 1].plot(range(1, N_SIMULATIONS + 1), r2_scores, marker='o', markersize=3, linestyle='-', color='#ff7f0e', alpha=0.8)
axes[0, 1].axhline(mean_r2, color='r', linestyle='--', label=f'Mean: {mean_r2:.3f}')
axes[0, 1].fill_between(range(1, N_SIMULATIONS + 1), mean_r2 - 2*std_r2, mean_r2 + 2*std_r2, color='r', alpha=0.1, label='2$\sigma$ Range')
axes[0, 1].set_xlabel('模拟次数')
axes[0, 1].set_ylabel('R² Score')
axes[0, 1].set_title('模拟过程中的 R² 波动')
axes[0, 1].legend()

# 图3：R² 分布概率密度 (KDE)
sns.histplot(r2_scores, kde=True, ax=axes[1, 0], color='purple', stat='density')
axes[1, 0].set_title('R² 分布概率密度 (KDE)')
axes[1, 0].set_xlabel('R²')

# 图4：特征重要性热力图 (Top 10)
feat_names = X_train_raw.columns
feat_mean = np.mean(feature_importances_matrix, axis=0)
feat_std = np.std(feature_importances_matrix, axis=0)
feat_cv = feat_std / (feat_mean + 1e-9)

feat_df = pd.DataFrame({
    'Feature': feat_names,
    'Importance Mean': feat_mean,
    'Importance Std': feat_std,
    'CV': feat_cv
}).sort_values('Importance Mean', ascending=False)

top_10_feats = feat_df.head(10)
heatmap_data = top_10_feats.set_index('Feature')[['Importance Mean', 'Importance Std']]
# 归一化
heatmap_data_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

sns.heatmap(heatmap_data_norm, annot=heatmap_data.values, fmt=".3f", cmap="YlOrRd", ax=axes[1, 1])
axes[1, 1].set_title('Top 10 特征重要性稳定性 (排列重要性)')

plt.suptitle(f'TabPFN 蒙特卡洛模拟评估报告 (N={N_SIMULATIONS}, Noise $\sigma$={NOISE_SIGMA})', fontsize=16)

# ==========================================
# 6. 结果保存
# ==========================================
summary_df = pd.DataFrame({
    'Metric': ['R2', 'MSE', 'RMSE', 'MAE'],
    'Mean': [mean_r2, mean_mse, np.mean(simulation_results['rmse']), np.mean(simulation_results['mae'])],
    'Std': [std_r2, std_mse, np.std(simulation_results['rmse']), np.std(simulation_results['mae'])],
    'CV(%)': [cv_r2, (std_mse/mean_mse)*100, (np.std(simulation_results['rmse'])/np.mean(simulation_results['rmse']))*100, (np.std(simulation_results['mae'])/np.mean(simulation_results['mae']))*100],
    'CI_Lower_95': [ci_r2[0], stats.t.interval(0.95, N_SIMULATIONS-1, loc=mean_mse, scale=stats.sem(mse_scores))[0], 0, 0],
    'CI_Upper_95': [ci_r2[1], stats.t.interval(0.95, N_SIMULATIONS-1, loc=mean_mse, scale=stats.sem(mse_scores))[1], 0, 0]
})

print("\n保存结果中...")
output_path = 'D:\研究生学习\博士学习\论文3\数据集\模型参数\PFN盆地all_MonteCarlo.xlsx'
try:
    with pd.ExcelWriter(output_path) as writer:
        summary_df.to_excel(writer, sheet_name='统计汇总', index=False)
        feat_df.to_excel(writer, sheet_name='特征重要性分析', index=False)
        # 保存所有迭代的详细指标，包括R2和RMSE
        all_iterations_df = pd.DataFrame({
            'R2': simulation_results['r2'],
            'RMSE': simulation_results['rmse'],
            'MSE': simulation_results['mse'],
            'MAE': simulation_results['mae']
        })
        all_iterations_df.to_excel(writer, sheet_name='所有迭代记录', index=False)
    print(f"结果已保存至: {output_path}")
except Exception as e:
    print(f"保存Excel失败: {e}")

plt.show()


