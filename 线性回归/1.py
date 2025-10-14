import numpy as np
import matplotlib.pyplot as plt

# 训练数据：输入特征x和对应的标签y
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# 前向传播函数：计算预测值 y = w * x
def forward(x):
    return x * w


# 损失函数：计算单个样本的平方误差损失
def loss(x, y):
    y_pred = forward(x)  # 使用当前权重w进行预测
    return (y_pred - y) ** 2  # 返回平方误差


# 初始化列表，用于存储权重和对应的均方误差
w_list = []
mse_list = []

# 遍历一系列可能的权重值，从0.0到40.0，步长为0.1
for w in np.arange(0.0, 41, 0.1):
    print("w=", w)
    loss_sum = 0  # 初始化当前权重下的总损失

    # 遍历所有训练样本
    for x_val, y_val in zip(x_data, y_data):
        # 计算当前样本的预测值
        y_pred_val = forward(x_val)
        # 计算当前样本的损失值
        loss_val = loss(x_val, y_val)
        # 累加损失值
        loss_sum += loss_val
        # 打印当前样本的详细信息
        print("\t", x_val, y_val, y_pred_val, loss_val)

    # 计算当前权重下的均方误差（MSE）
    mse = loss_sum / 3
    print("MSE=", mse)

    # 存储当前权重和对应的均方误差
    w_list.append(w)
    mse_list.append(mse)

# 可选：绘制权重与MSE的关系图
# plt.plot(w_list, mse_list)
# plt.xlabel('Weight (w)')
# plt.ylabel('Mean Squared Error (MSE)')
# plt.title('Weight vs MSE')
# plt.show()
