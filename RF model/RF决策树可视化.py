import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt  # 新增matplotlib导入
import graphviz
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
from six import StringIO
from IPython.display import Image

# 从Excel文件中读取数据
data= pd.read_excel('D:\研究生学习\博士学习\论文3\数据集\工业分析元素分析焦油产率数据收集.xlsx', sheet_name="汇总")

# 提取特征和标签
X = data.iloc[:, 1:7]  # 提取后6列作为特征
y = data.iloc[:, 0]    # 第一列作为标签

# 构建随机森林回归模型
model = RandomForestRegressor(
    n_estimators=50,
    random_state=42, 
    max_depth = 20,
    min_samples_leaf = 4,
    min_samples_split = 2,
    ccp_alpha = 0.1,
    bootstrap = True)

model.fit(X, y)

# 选择一棵决策树进行可视化
tree_to_visualize = model.estimators_[0]

# 设置画布大小
plt.figure(figsize=(16, 12))

# 可视化决策树
plot_tree(tree_to_visualize, 
          filled=True, 
          rounded=True, 
          feature_names=X.columns,
          proportion=True,
          fontsize=8)

# 保存可视化结果
save_path = r'D:\研究生学习\博士学习\论文3\绘图\random_forest.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"可视化结果已保存到'{save_path}'")
