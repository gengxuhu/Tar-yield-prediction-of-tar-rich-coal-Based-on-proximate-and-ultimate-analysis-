# 导入必要的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from scipy import stats
import matplotlib
import warnings
import re
import shap
import os
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.neural_network import MLPRegressor
import matplotlib.colors as mcolors
import time  # 导入time模块
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林回归模型
matplotlib.use('TkAgg')

# 记录程序开始时间
start_time = time.time()


config = {
    "font.family": 'serif',
    "font.size": 16,  # 相当于小四大小
    "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ['Times New Roman'],  # 宋体
    'axes.unicode_minus': False  # 处理负号，即-号
}
rcParams.update(config)

# 读取Excel数据

warnings.filterwarnings("ignore", category=RuntimeWarning)  # 忽略特定类型的运行时警告，避免不必要的输出


def sanitize_filename(name):  # 定义一个函数，用于清理文件名中的非法字符    
    return re.sub(r'[\\/*?:"<>|]', '_', name)  # 使用正则表达式将Windows文件名中的非法字符替换为下划线


excel_path = r'D:\研究生学习\博士学习\论文3\数据集\工业分析元素分析焦油产率数据归一化2.xlsx'  # 替换为实际文件路径
train_data = pd.read_excel(excel_path, sheet_name='Atrain')
test_data = pd.read_excel(excel_path, sheet_name='Atest')

# 提取特征和目标变量69
X_train = train_data.loc[:, "A":"N"]  # 特征列从M到S
y_train = train_data["Tar"]
X_test = test_data.loc[:, "A":"N"]
y_test = test_data["Tar"]

# 在文件开头添加
industrial_analysis_labels = {
    'M': '$\mathregular{M_a}$$\mathregular{_d}$',
    'A': '$\mathregular{A_d}$',
    'V': '$\mathregular{V_d}$$\mathregular{_a}$$\mathregular{_f}$',
    'FC': '$\mathregular{FC_a}$$\mathregular{_d}$',
    'C': '$\mathregular{C_d}$$\mathregular{_a}$$\mathregular{_f}$',
    'H': '$\mathregular{H_d}$$\mathregular{_a}$$\mathregular{_f}$',
    'O': '$\mathregular{O_d}$$\mathregular{_a}$$\mathregular{_f}$',
    'N': '$\mathregular{N_d}$$\mathregular{_a}$$\mathregular{_f}$',
    'S': '$\mathregular{S_t}$$\mathregular{_,}$$\mathregular{_d}$'}



feature_names = X_train.columns.tolist()
# 建立MLP回归模型
model = MLPRegressor(
    hidden_layer_sizes=(100, 50,30),  # 增加隐藏层数量和神经元数量，增强模型复杂度
    activation='relu',  # 激活函数，'relu' 表示使用修正线性单元函数，用于在神经元中引入非线性特性
    solver='lbfgs',  # 权重优化的求解器，'adam' 是一种基于自适应矩估计的优化算法，适用于较大数据集和较高维度的问题
    alpha=0.001,  # 减小L2正则化项参数，降低正则化强度，帮助模型更好拟合数据
    batch_size='auto',  # 训练模型时的批次大小，'auto' 表示批次大小为min(200, n_samples)
    learning_rate='constant',  # 学习率策略，'adaptive' 表示当训练损失在连续两次迭代中没有改善时，学习率会自动降低
    max_iter=600,  # 增加最大迭代次数，让模型有更多机会学习数据特征
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
rmse = np.sqrt(mse)
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
train_results = pd.DataFrame({'真实Y值': y_train.values, '预测Y值': x_pred})
test_results = pd.DataFrame({'真实Y值': y_test.values, '预测Y值': y_pred})

# 将DataFrame写入Excel文件
with pd.ExcelWriter(r'D:\研究生学习\博士学习\论文3\数据集\模型参数\MLPa.xlsx') as writer:
    train_results.to_excel(writer, sheet_name='train', index=False)
    test_results.to_excel(writer, sheet_name='test', index=False)
    evaluation_df.to_excel(writer, sheet_name='评估结果', index=False)

# SHAP分析准备

print("\n正在使用最佳模型在*测试集*上计算SHAP值...")  # 打印开始在测试集上计算SHAP值的提示

# 对于scikit-learn的MLP模型，需要使用masker参数
explainer = shap.Explainer(model.predict, X_train)  # 使用训练好的最佳模型创建一个SHAP解释器

print("正在计算主效应SHAP值（基于X_train）")  # 打印开始计算主效应SHAP值的提示

shap_values_obj = explainer(X_train)  # 将测试集数据传入解释器，计算每个样本每个特征的SHAP值
shap_values = shap_values_obj.values  # 从SHAP值对象中提取SHAP值数组矩阵
print("主效应SHAP值计算完成")  # 打印主效应SHAP值计算完成的提示

print("\n计算交互效应SHAP值（基于X_train）")  # 打印开始计算交互效应SHAP值的提示
# 对于ExactExplainer，需要使用不同的方法计算交互效应
# shap_interaction_values = explainer.shap_interaction_values(X_train)  # 这个方法不适用于ExactExplainer
print("当前SHAP版本中ExactExplainer不支持交互效应计算，跳过交互效应分析。")
shap_interaction_values = None  # 设置为None以避免后续错误

print("\n正在根据SHAP值对特征进行排序……")
mean_shap = np.abs(shap_values).mean(axis=0)  # 计算每个特征的SHAP绝对值的平均值，作为其重要性度量
shap_df = pd.DataFrame({  # 创建一个DataFrame来存储特征名和其重要性
    "feature": feature_names,  # 特征名称列
    "mean_shap": mean_shap  # 评价绝对SHAP值列
}).sort_values("mean_shap", ascending=False)  # 按重要性进行排序
sorted_features = shap_df["feature"].values  # 获取排序后的特征名称列表
print("特征排序完成。")  # 打印排序完成的提示

X_train_sorted = X_train[sorted_features]  # 按照排序后的特征顺序，重新排列测试集的列
orig_index = [feature_names.index(f) for f in sorted_features]  # 获取排序后特征在原始特征列表中的索引
shap_values_sorted = shap_values[:, orig_index]  # 按照新的特征顺序，重新排列SHAP值矩阵的列
# shap_interaction_values_sorted = shap_interaction_values[:, orig_index][:, :, orig_index]  # 由于交互效应不可用，注释掉这行


# 通用辅助函数
def bootstrap_lowess_ci(x, y, n_boot=200, frac=0.5, ci_level=0.95):  # 定义一个函数，用bootstrap方法计算lowess平滑的置信区间
    """ 使用bootstrap方法计算lowess平滑的置信区间。
      参数说明:  x (pd.Series): 模型的输入特征（自变量）。
                 y (pd.Series): 模型的输出或真实值（因变量）。
                 n_boot (int): bootstrap抽样的次数。次数越多，置信区间的估计越稳定，但计算成本也越高。默认为200次。
                 frac (float): lowess平滑器中使用的样本比例。这个值控制平滑的程度，介于0和1之间。 值越小，曲线越贴近数据点；值越大，曲线越平滑。默认为0.5。
                 ci_level (float): 置信区间的水平。例如，0.95表示计算95%的置信区间。默认为0.95。
                 返回:
                 tuple: (主平滑曲线, (x轴范围, 置信下界, 置信上界)) 或 (None, None)"""

    if len(x) < 10: return None, None  # 如果样本点太少，则不进行计算
    boot_lines = []  # 初始化一个列表，用于存储每次bootstrap抽样得到的平滑曲线
    x_range = np.linspace(x.min(), x.max(), 100)  # 在x的范围内生成100个等间距点，用于插值
    for _ in range(n_boot):  # 循环进行n_boot次bootstrap抽样
        sample_indices = np.random.choice(len(x), len(x), replace=True)  # 有放回地抽取样本索引
        x_sample, y_sample = x.iloc[sample_indices], y[sample_indices]  # 根据索引获取抽样数据
        sorted_indices = np.argsort(x_sample)  # 对抽样的x值进行排序
        x_sorted, y_sorted = x_sample.iloc[sorted_indices].values, y_sample[sorted_indices]  # 获取排序后的x和y
        if len(np.unique(x_sorted)) < 2: continue  # 如果抽样后x的唯一值少于2个，则跳过此次循环
        smoothed = lowess(y_sorted, x_sorted, frac=frac)  # 对抽样数据进行lowess平滑
        interp_func = np.interp(x_range, smoothed[:, 0], smoothed[:, 1])  # 将平滑结果插值到x_range上
        boot_lines.append(interp_func)  # 将插值后的曲线添加到列表中
        if not boot_lines: return None, None  # 如果未能生成任何bootstrap曲线，则返回None
        sorted_indices_orig = np.argsort(x)  # 对原始x数据进行排序
        x_sorted_orig, y_sorted_orig = x.iloc[sorted_indices_orig].values, y[sorted_indices_orig]  # 获取排序后的原始x和y
        main_smoothed = lowess(y_sorted_orig, x_sorted_orig, frac=frac)  # 对完整的原始数据进行lowess平滑，作为主曲线
        boot_lines_arr = np.array(boot_lines)  # 将bootstrap曲线列表转换为numpy数组
        alpha = (1 - ci_level) / 2  # 计算置信水平对应的alpha值
        lower_bound, upper_bound = np.quantile(boot_lines_arr, alpha, axis=0), np.quantile(boot_lines_arr, 1 - alpha,
                                                                                           axis=0)  # 计算每个点的上下置信边界
        return main_smoothed, (x_range, lower_bound, upper_bound)  # 返回主平滑曲线和置信区间数据


def find_and_plot_crossings(ax, x_curve, y_curve, color, base_y_offset=0.9):  # 定义一个函数，用于寻找并绘制曲线与y=0的交点（阈值）
    """在给定的Matplotlib Axes上寻找并绘制一条曲线与y=0轴的交点（阈值）。
       该函数通过线性插值精确计算交点位置，并用垂直虚线和文本标签在图上标记出来。
       文本标签会自动上下交错排列以避免重叠。
       参数说明:
       ax (matplotlib.axes.Axes): 要在其上绘图的Matplotlib子图对象。
       x_curve (np.array): 曲线的x坐标数组。
       y_curve (np.array): 曲线的y坐标数组。函数将寻找此曲线与y=0的交点。
       color (str): 用于绘制垂直线和文本背景的颜色。应与对应曲线的颜色匹配。
       base_y_offset (float): 控制文本标签垂直位置的基准偏移量，相对于y轴的高度。
                              默认为0.9，即从顶部向下10%的位置开始，然后交替向下排列。 """
    sign_changes = np.where(np.diff(np.sign(y_curve)))[0]  # 找到y值符号发生变化的位置
    for i, k in enumerate(sign_changes):  # 遍历所有符号变化点
        x1, y1, x2, y2 = x_curve[k], y_curve[k], x_curve[k + 1], y_curve[k + 1]  # 获取变化点前后的坐标
        if (y2 - y1) == 0: continue  # 避免除以零
        x_root = x1 - y1 * (x2 - x1) / (y2 - y1)  # 使用线性插值计算交点的x坐标（根）
        ax.axvline(x=x_root, color=color, linestyle='--', linewidth=1)  # 在交点位置绘制一条垂直虚线
        y_text_position = ax.get_ylim()[1] * (base_y_offset - (i % 2) * 0.1)  # 计算文本标签的y坐标，使其上下交错防止重叠
        ax.text(x_root, y_text_position, f' {x_root:.2f} ', color='white', backgroundcolor=color, ha='center',
                va='center', fontsize=16, bbox=dict(facecolor=color, edgecolor='none', pad=1))  # 在垂直线上方添加文本标签显示交点值


def find_roots(x_curve, y_curve):  # 定义一个函数，只计算曲线与y=0的交点，不绘图    
    """ 计算一条曲线与y=0轴的所有交点（即方程的根），但不进行绘图。
    该函数通过线性插值法来精确估算交点的x坐标值。
    参数说明:  x_curve (np.array): 曲线的x坐标数组。
            y_curve (np.array): 曲线的y坐标数组。函数将寻找此曲线与y=0的交点。
            返回:
            list: 一个包含所有计算出的交点（根）的x坐标值的列表。如果不存在交点，则返回空列表。"""
    roots = []
    sign_changes = np.where(np.diff(np.sign(y_curve)))[0]  # 找到y值符号发生变化的位置
    for k in sign_changes:  # 遍历所有符号变化点
        x1, y1, x2, y2 = x_curve[k], y_curve[k], x_curve[k + 1], y_curve[k + 1]  # 获取变化点前后的坐标
        if (y2 - y1) == 0: continue  # 避免除以零
        x_root = x1 - y1 * (x2 - x1) / (y2 - y1)  # 使用线性插值计算交点的x坐标
        roots.append(x_root)  # 将计算出的交点添加到列表中
    return roots


# SHAP总览图与依赖图绘制
print("\n正在绘制 SHAP 特征重要性总览图...")

# 创建图形和坐标轴
fig = plt.figure(figsize=(10, 10), dpi=300)
ax_sw = fig.add_axes([0.32, 0.11, 0.59, 0.8])  # 蜂群图坐标轴
ax_bar = ax_sw.twiny()  # 条形图坐标轴

# 设置坐标轴层级和透明度
ax_bar.set_zorder(0)
ax_sw.set_zorder(1)
ax_sw.patch.set_alpha(0)

# 计算y轴位置并绘制条形图
y_pos = np.arange(len(sorted_features))[::-1]
ax_bar.barh(y=y_pos, width=shap_df["mean_shap"].values, height=0.7, 
            color="blue", alpha=0.5, edgecolor="none", zorder=0)

# 配置条形图x轴
xlim_bar = shap_df["mean_shap"].values.max() * 1.05
ax_bar.set_xlim(0, xlim_bar)
xticks_bar = np.linspace(0, xlim_bar, 5)
ax_bar.set_xticks(xticks_bar)
ax_bar.set_xticklabels([f"{x:.2f}" for x in xticks_bar], fontsize=34)
ax_bar.set_xlabel("Mean (|SHAP| value)", fontsize=34)
ax_bar.set_yticks(y_pos)
ax_bar.tick_params(axis='x', direction='in', length=4)

# 配置蜂群图x轴
max_abs_shap = np.abs(shap_values_sorted).max()
xlim_sw = max_abs_shap * 1.1
ax_sw.set_xlim(-xlim_sw, xlim_sw)
sw_xticks = np.linspace(-xlim_sw, xlim_sw, 5)
ax_sw.set_xticks(sw_xticks)
ax_sw.set_xticklabels([f"{x:.2f}" for x in sw_xticks], fontsize=34)
ax_sw.set_xlabel("Relative importance", fontsize=34)

# 统一设置刻度参数
ax_sw.tick_params(axis='both', direction='in', length=4, labelsize=20)

# 创建SHAP解释对象并绘制蜂群图
expl_main = shap.Explanation(
    values=shap_values_sorted,
    data=X_train_sorted.values,
    feature_names=list(sorted_features),
    base_values=shap_values_obj.base_values[0]
)
shap.plots.beeswarm(expl_main, max_display=len(sorted_features), 
                   ax=ax_sw, show=False, plot_size=None)

# 手动覆盖蜂群图x轴字体大小与标题
ax_sw.set_xlabel("Relative importance", fontsize=34)
ax_sw.tick_params(axis='x', labelsize=34)

# 设置y轴标签
ax_sw.set_yticks(y_pos)
ax_sw.set_yticklabels([industrial_analysis_labels.get(f, f) for f in sorted_features], fontsize=34)

# 美化图形：移除边框
for ax in [ax_sw, ax_bar]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 添加标注并保存图形
ax_sw.text(0.02, 0.98, '(j)', transform=ax_sw.transAxes,
           fontsize=34, va='top', ha='left', fontname='Times New Roman')

# 获取蜂群图的颜色映射条并设置字体大小
# shap.plots.beeswarm 内部会生成一个 colorbar，可通过 fig.axes 找到
# 通常最后一个 axes 是 colorbar
if len(fig.axes) > 2:
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(labelsize=34)   # 设置 colorbar 刻度字体大小
    cbar_ax.set_ylabel('', fontsize=34)  # 设置 colorbar 标签字体大小

output_main_folder = r"D:\研究生学习\博士学习\论文3\绘图\shapMLP\A"
combined_image_path = os.path.join(output_main_folder, 'combined_shap_summary_plot.svg')
plt.savefig(combined_image_path, dpi=300, bbox_inches='tight')
print(f"SHAP 特征重要性总览图已保存至: '{combined_image_path}'")
#plt.show()
# 交互作用图绘制
print("\n正在绘制 SHAP 交互作用总览图...")
plt.figure()
# shap.summary_plot(shap_interaction_values_sorted, X_train_sorted, 
#                  max_display=10, show=False)
print("  -> 注意: 当前SHAP版本中的ExactExplainer不支持交互作用值计算，已跳过交互作用总览图")

interaction_summary_plot_path = os.path.join(output_main_folder, 'shap_interaction_summary_plot.png')
plt.savefig(interaction_summary_plot_path, dpi=300, bbox_inches='tight')
print(f"SHAP 交互作用总览图已保存至: '{interaction_summary_plot_path}'")


def plot_shap_dependence(feature_name, x_values, shap_values_for_feature, save_folder,
                         custom_annotation=None):  # 定义一个函数，用于绘制单个特征的SHAP依赖图

    """ 绘制并保存单个特征的SHAP依赖图。
      **SHAP值散点图**: 显示每个样本的特征值与其对应的SHAP值的关系（蓝色散点）。
      **特征值分布直方图**: 以背景条形图的形式展示该特征在数据集中的分布情况（红色条形）。
      **LOWESS平滑拟合曲线**: 揭示SHAP值随特征值变化的平均趋势（深蓝色实线）。
      **置信区间**: 为LOWESS曲线提供统计可靠性范围，通常是95%置信区间（浅蓝色填充区域）。
      **阈值（交点）标定**: 自动寻找并标记出拟合曲线与y=0的交点，这些点是特征影响方向（正/负）发生改变的关键阈值（绿色虚线和标签）。
      **自定义注释**: 允许用户在图上添加自定义文本。
      参数说明:
      feature_name (str): 要绘制的特征的名称。将用作图表X轴标签和输出文件名的一部分。
      x_values (pd.Series or np.array): 该特征在数据集中的所有样本值。
      shap_values_for_feature (np.array): 与x_values一一对应的SHAP值。
      save_folder (str): 用于保存生成图像文件的文件夹路径。
      custom_annotation (dict, optional): 一个可选字典，用于在图上添加自定义注释。
      例如: {'text': '关键区域', 'x': 0.8, 'y': 0.8}
    """

    print(f"  -> 正在绘制特征: {feature_name} ...")  # 打印正在绘制哪个特征的提示
    fig_dep, ax1 = plt.subplots(figsize=(8, 6), dpi=150)  # 创建一个新的图形和子图
    ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
    ax2.patch.set_alpha(0)  # 将第二个y轴的背景设置为透明
    counts, bin_edges = np.histogram(x_values, bins=30)  # 计算特征值的分布直方图数据
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # 计算每个柱子的中心位置
    bin_width = bin_edges[1] - bin_edges[0]  # 计算每个柱子的宽度
    ax1.bar(bin_centers, counts, width=bin_width * 0.6, align='center', color='#4B0082', alpha=0.3,
            label='Distribution')  # 在ax1上绘制分布直方图
    ax1.set_ylabel('Distribution', fontsize=18)  # 设置ax1的y轴标签
    ax1.set_ylim(0, counts.max() * 1.1)  # 设置ax1的y轴范围
    ax2.scatter(x_values, shap_values_for_feature, alpha=0.3, s=25, color='#00008B', label='Sample',
                zorder=2)  # 在ax2上绘制SHAP值的散点图
    if len(x_values) > 1:  # 检查样本数量是否足够
        main_fit, ci_data = bootstrap_lowess_ci(x_values, shap_values_for_feature,
                                                frac=0.3)  # 计算lowess平滑曲线和置信区间
        if main_fit is not None and ci_data is not None:  # 如果成功计算
            ax2.plot(main_fit[:, 0], main_fit[:, 1], color='#9400D3', lw=2, label='LOWESS Fit',
                     zorder=4)  # 绘制主平滑曲线
            ax2.fill_between(ci_data[0], ci_data[1], ci_data[2], color='#9400D3', alpha=0.1,
                             label='95%CI')  # 填充置信区间，使用更醒目的颜色和适当透明度
            ax2.axhline(0, color='black', linestyle='--', lw=1, zorder=1)  # 绘制y=0的参考线
            find_and_plot_crossings(ax2, main_fit[:, 0], main_fit[:, 1], 'black')  # 寻找并绘制阈值线
        ax2.set_ylabel('SHAP value', fontsize=18)  # 设置ax2的y轴标签
        y_max = np.abs(shap_values_for_feature).max() * 1.15  # 计算ax2的y轴范围
        if y_max < 1e-6: y_max = 1  # 避免范围过小
        ax2.set_ylim(-y_max, y_max)  # 设置ax2的y轴范围
        # 设置X轴标签：使用工业分析标签映射，若不存在则保持原名称
        x_label = industrial_analysis_labels.get(feature_name, feature_name)
        ax1.set_xlabel(x_label, fontsize=18)  # 设置共享的x轴标签
        # 设置坐标轴刻度线朝向内
        ax1.tick_params(direction='in')
        ax2.tick_params(direction='in')
        if custom_annotation and isinstance(custom_annotation, dict):  # 检查是否有自定义注释
            text = custom_annotation.get('text', '')
            x_pos = custom_annotation.get('x', 0.95)
            y_pos = custom_annotation.get('y', 0.95)
            props = {'ha': custom_annotation.get('ha', 'right'), 'va': custom_annotation.get('va', 'top'),
                     'fontsize': custom_annotation.get('fontsize', 12),
                     'color': custom_annotation.get('color', 'darkred'),
                     'fontweight': custom_annotation.get('fontweight', 'bold')}
            ax1.text(x_pos, y_pos, text, transform=ax1.transAxes, **props)  # 在图上添加自定义注释
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h2 + h1, l2 + l1, loc='upper right', fontsize=14)  # 合并两个y轴的图例并显示
        sanitized_feature_name = sanitize_filename(feature_name)  # 清理特征名以用作文件名
        output_filename = f"dependence_plot_{sanitized_feature_name}.svg"  # 构建输出文件名
        full_path = os.path.join(save_folder, output_filename)  # 拼接完整保存路径
        fig_dep.savefig(full_path, dpi=300, bbox_inches='tight')  # 保存图形
        plt.close(fig_dep)  # 关闭图形，释放内存


print("\n开始为所有特征批量生成依赖图...")  # 打印批量生成依赖图的开始提示
output_folder_dep = os.path.join(output_main_folder, 'dependence_plots')  # 定义依赖图的输出文件夹路径
os.makedirs(output_folder_dep, exist_ok=True)  # 创建该文件夹
print(f"所有依赖图将被保存到 '{output_folder_dep}' 文件夹中。")  # 打印保存路径提示
for i, feature_name in enumerate(sorted_features):  # 遍历所有排序后的特征
    x_data_loop = X_train_sorted[feature_name]  # 获取当前特征的数值
    if not np.isfinite(x_data_loop).all():  # 检查特征值中是否包含NaN或无穷大等非有限值
        print(f"  -> 跳过特征: '{feature_name}'，因为它包含非有限值 (例如 NaN)。")  # 如果有，则跳过该特征
        continue
    y_data_shap_loop = shap_values_sorted[:, i]  # 获取当前特征对应的SHAP值   
    annotation_for_this_plot = None  # 初始化注释为空
    plot_shap_dependence(feature_name=feature_name, x_values=x_data_loop, shap_values_for_feature=y_data_shap_loop,
                         save_folder=output_folder_dep, custom_annotation=annotation_for_this_plot)  # 调用绘图函数绘制并保存依赖图
print(f"\n任务完成！所有依赖图已成功生成并保存。")  # 打印任务完成的提示

# SHAP交互图绘制
print("\n--- 开始执行高级交互图绘制任务 (最终修正版) ---")  # 打印高级绘图任务开始的提示


# 定义一个函数，用于绘制两个特征之间的高级交互图
def plot_advanced_interaction(primary_feature_name, interacting_feature_name, x_values, interaction_feature_values,
                              shap_interaction_slice, save_folder):
    """
    绘制并保存一个高级的、信息丰富的特征交互SHAP图。
    该函数旨在可视化一个主特征（primary_feature）的SHAP值如何受到一个交互特征（interacting_feature）的影响。
    图表主要包含以下几个部分：
    1.  **交互散点图**: 以主特征值为X轴，SHAP交互值为Y轴。散点的颜色由交互特征的值决定， 
        使用'seismic'（蓝-白-红）色谱，直观展示交互特征值高低对主特征效应的影响。
    2.  **分组拟合曲线**: 将交互特征按其中位数分为“高值组”和“低值组”，并为这两组数据
        分别绘制LOWESS平滑拟合曲线（红色和蓝色实线）及其置信区间。这清晰地揭示了
        主特征的效应趋势是否因交互特征的水平不同而改变。
    3.  **共同阈值标定**: 自动计算并寻找两组拟合曲线共同穿过y=0的“稳定”阈值点。
        如果找到，则用紫色虚线和标签在图上标出。这个阈值点可能代表一个不受交互特征影响的、
        稳健的效应转变点。
    4.  **背景分布图**: 以灰色条形图在背景中展示主特征的数据分布，为趋势分析提供数据密度参考。
    参数说明:
    primary_feature_name (str): 主特征的名称，将显示在X轴。
    interacting_feature_name (str): 交互特征的名称，其值将决定散点的颜色和分组。
    x_values (pd.Series or np.array): 主特征的实际值数组。
    interaction_feature_values (pd.Series or np.array): 交互特征的实际值数组。
    shap_interaction_slice (np.array): 对应的主特征与交互特征之间的SHAP交互值数组。
    save_folder (str): 用于保存生成图像文件的文件夹路径。

    """
    sanitized_primary, sanitized_interacting = sanitize_filename(primary_feature_name), sanitize_filename(
        interacting_feature_name)  # 清理主特征和交互特征的名称
    print(f"  -> 正在绘制: '{primary_feature_name}' (交互特征: '{interacting_feature_name}')")  # 打印正在绘制哪对特征的提示
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=150)  # 创建一个新的图形和子图
    ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴

    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["blue", "#4B0082", "red"])
    points = ax2.scatter(x_values, shap_interaction_slice, c=interaction_feature_values,
                         cmap=cmap, alpha=1, s=25, zorder=2, label='sample')  # 在ax2上绘制散点图，点的颜色由交互特征的值决定
    median_val = interaction_feature_values.median()  # 计算交互特征值的中位数
    low_mask, high_mask = interaction_feature_values <= median_val, interaction_feature_values > median_val  # 根据中位数创建两个布尔掩码，用于分组
    # --- 修改1：修正图例(Legend)的标签文本 ---   
    groups = {  # 定义两个组的配置信息
        'low': {'mask': low_mask, 'color': 'blue', 'offset': 0.9,
                'label': f' {interacting_feature_name} <= {median_val:.2f}'},  # 低值组的配置，包含动态生成的图例标签        
        'high': {'mask': high_mask, 'color': 'red', 'offset': 0.8,
                 'label': f' {interacting_feature_name} > {median_val:.2f}'}  # 高值组的配置   
    }
    counts, bin_edges = np.histogram(x_values, bins=30)  # 计算主特征的分布数据   
    bin_centers, bin_width = (bin_edges[:-1] + bin_edges[1:]) / 2, bin_edges[1] - bin_edges[0]  # 计算柱子中心和宽度   
    ax1.bar(bin_centers, counts, width=bin_width * 0.7, color='gray', alpha=0.2,
            label='Distribution')  # 在ax1上绘制分布直方图   
    ax1.set_ylabel('Distribution', fontsize=12)  # 设置ax1的y轴标签   
    ax1.set_ylim(0, counts.max() * 1.1)  # 设置ax1的y轴范围   
    fits, roots = {}, {}  # 初始化字典用于存储拟合曲线和根

    for name, info in groups.items():  # 遍历高低值两个组
        x_group, shap_group = x_values[info['mask']], shap_interaction_slice[info['mask']]  # 根据掩码获取分组数据        
        if len(x_group) < 10: continue  # 如果组内样本太少，则跳过 
        main_fit, ci_data = bootstrap_lowess_ci(x_group, shap_group)  # 对该组数据进行lowess平滑和置信区间计算        
        if main_fit is not None and ci_data is not None:  # 如果计算成功           
            ax2.plot(main_fit[:, 0], main_fit[:, 1], color=f'dark{info["color"]}', lw=2.5,
                     label=info['label'])  # <-- 使用修正后的标签来绘制平滑曲线
            ax2.fill_between(ci_data[0], ci_data[1], ci_data[2], color=info['color'], alpha=0.15)  # 填充该曲线的置信区间
            fits[name] = main_fit  # 存储拟合曲线
            roots[name] = find_roots(main_fit[:, 0], main_fit[:, 1])  # 计算并存储该曲线的根    
    if 'low' in roots and 'high' in roots:  # 如果高低两组都找到了根
        tolerance = (x_values.max() - x_values.min()) * 0.05  # 定义一个容差，用于判断两个根是否“接近”
        for r_low in roots['low']:  # 遍历低值组的根
            for r_high in roots['high']:  # 遍历高值组的根
                if abs(r_low - r_high) < tolerance:  # 如果两个根非常接近 
                    avg_root = (r_low + r_high) / 2  # 计算它们的平均值
                    ax2.axvline(x=avg_root, color='black', linestyle='--', linewidth=1)  # 在该位置绘制一条紫色的垂直虚线，表示共同的阈值
                    ax2.text(avg_root, ax2.get_ylim()[1] * 0.9, f' {avg_root:.2f} ', color='white',
                             backgroundcolor='purple', ha='center', va='center', fontsize=9,
                             bbox=dict(facecolor='purple', edgecolor='none', pad=1))  # 在线上方添加文本标签

    # ---调整颜色条(Color Bar)的间距 ---    
    # 在图窗(Figure)的指定位置手动创建一个新的坐标系(Axes)，专门用于放置颜色条。    
    # 这种方法提供了最精确的布局控制。    
    # 坐标 [left, bottom, width, height] 是相对于整个图窗的比例（0到1）。    
    #   left=0.92: 颜色条的左边界从图窗左侧92%处开始，即放在主图的右侧并留出一些间隙。    
    #   bottom=0.11, height=0.77: 定义了颜色条的垂直位置和长度，使其与主图的坐标系在垂直方向上对齐，解决了颜色条自动变短的问题。    
    #   width=0.03: 定义了颜色条的宽度，这个值现在是唯一独立控制颜色条“粗细”的参数。
    cbar_ax = fig.add_axes([0.975, 0.11, 0.02, 0.77])
    # 将颜色条(colorbar)绘制到我们刚刚创建的专用坐标系 cbar_ax 中。
    #   points: 这是散点图对象(ax.scatter的返回值)，颜色条将根据它的颜色映射(cmap)和数据范围来绘制。
    #   cax=cbar_ax: 这里的 cax 参数是关键，它告诉matplotlib“请把颜色条画在这个指定的cbar_ax里”，而不是让它自动寻找位置。
    cbar = fig.colorbar(points, cax=cbar_ax)
    cbar.set_label(f"Value of {interacting_feature_name}", size=12)  # 设置颜色条的标签
    # 使用工业分析标签映射设置X轴标签
    ax1.set_xlabel(industrial_analysis_labels.get(primary_feature_name, primary_feature_name), fontsize=12)
    ax2.set_ylabel(f'SHAP Interaction Value', fontsize=12)  # 设置ax2的y轴标签
    # fig.suptitle(f"Interaction: {primary_feature_name} vs {interacting_feature_name}", fontsize=14) # 设置整个图的标题   
    ax2.axhline(0, color='black', linestyle='--', lw=1, zorder=0)  # 绘制y=0的参考线   
    y_max_abs = np.abs(shap_interaction_slice).max() * 1.1  # 计算ax2的y轴范围   
    ax2.set_ylim(-y_max_abs if y_max_abs > 1e-6 else -1, y_max_abs if y_max_abs > 1e-6 else 1)  # 设置ax2的y轴范围   
    ax2.legend(loc='best', fontsize=10)  # 显示图例   
    ax1.set_zorder(0);  # 将ax1置于底层   
    ax2.set_zorder(1);  # 将ax2置于顶层   
    ax2.patch.set_alpha(0)  # 将ax2的背景设置为透明   
    output_filename = f"interaction_{sanitized_primary}_vs_{sanitized_interacting}.png"  # 构建输出文件名   
    full_path = os.path.join(save_folder, output_filename)  # 拼接完整保存路径   
    fig.savefig(full_path, dpi=200, bbox_inches='tight')  # 保存图形   
    plt.close(fig)  # 关闭图形，释放内存


output_folder_advanced_interactions = os.path.join(output_main_folder,
                                                   'advanced_interaction_plots_final')  # 定义高级交互图的输出文件夹
os.makedirs(output_folder_advanced_interactions, exist_ok=True)  # 创建该文件夹
print(f"\n所有高级交互图将按指定条件被保存到 '{output_folder_advanced_interactions}' 文件夹中。")  # 打印保存路径提示
n_features = len(sorted_features)  # 获取特征总数

for i in range(n_features):  # 遍历所有特征，作为主特征   
    primary_feature_name = sorted_features[i]  # 获取主特征名称    
    for j in range(n_features):  # 再次遍历所有特征，作为交互特征        
        if i == j: continue  # 如果是同一个特征，则跳过       
        interacting_feature_name = sorted_features[j]  # 获取交互特征名称       
        x_values = X_train_sorted[primary_feature_name]  # 获取主特征的数值       
        interaction_feature_values = X_train_sorted[interacting_feature_name]  # 获取交互特征的数值（用于着色）       
        # shap_interaction_slice = shap_interaction_values_sorted[:, i,
        #                          j] * 2  # 由于交互效应不可用，注释掉这行
        print(f"  -> 跳过交互图: '{primary_feature_name}' vs '{interacting_feature_name}' (交互效应不可用)")
        continue  # 跳过交互图绘制       
        plot_advanced_interaction(  # 调用高级交互图的绘图函数           
            primary_feature_name=primary_feature_name,
            interacting_feature_name=interacting_feature_name,
            x_values=x_values,
            interaction_feature_values=interaction_feature_values,
            shap_interaction_slice=shap_interaction_slice,
            save_folder=output_folder_advanced_interactions
        )
print(f"\n--- 所有绘图任务完成！ ---")  # 打印所有任务完成的最终提示











