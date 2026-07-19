
# 示例：如何在其他程序中加载缩放器并进行逆变换
import pickle
import pandas as pd

# 加载保存的缩放器
scaler_file = 'minmax_scaler.pkl'  # 替换为实际路径
with open(scaler_file, 'rb') as f:
    scaler = pickle.load(f)

# 读取归一化后的数据
normalized_data = pd.read_excel('归一化数据.xlsx')  # 替换为实际文件

# 选择需要逆变换的列
columns_to_denormalize = ['M', 'A', 'V', 'FC', 'C', 'H', 'O', 'N', 'S']
normalized_values = normalized_data[columns_to_denormalize]

# 执行逆变换
original_values = pd.DataFrame(
    scaler.inverse_transform(normalized_values),
    columns=columns_to_denormalize,
    index=normalized_data.index
)

# 现在 original_values 包含了逆变换后的原始数据
print(original_values)
