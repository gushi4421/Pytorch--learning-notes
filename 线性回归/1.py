import numpy as np
import matplotlib.pyplot as plt

# ѵ�����ݣ���������x�Ͷ�Ӧ�ı�ǩy
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# ǰ�򴫲�����������Ԥ��ֵ y = w * x
def forward(x):
    return x * w


# ��ʧ���������㵥��������ƽ�������ʧ
def loss(x, y):
    y_pred = forward(x)  # ʹ�õ�ǰȨ��w����Ԥ��
    return (y_pred - y) ** 2  # ����ƽ�����


# ��ʼ���б����ڴ洢Ȩ�غͶ�Ӧ�ľ������
w_list = []
mse_list = []

# ����һϵ�п��ܵ�Ȩ��ֵ����0.0��40.0������Ϊ0.1
for w in np.arange(0.0, 41, 0.1):
    print("w=", w)
    loss_sum = 0  # ��ʼ����ǰȨ���µ�����ʧ

    # ��������ѵ������
    for x_val, y_val in zip(x_data, y_data):
        # ���㵱ǰ������Ԥ��ֵ
        y_pred_val = forward(x_val)
        # ���㵱ǰ��������ʧֵ
        loss_val = loss(x_val, y_val)
        # �ۼ���ʧֵ
        loss_sum += loss_val
        # ��ӡ��ǰ��������ϸ��Ϣ
        print("\t", x_val, y_val, y_pred_val, loss_val)

    # ���㵱ǰȨ���µľ�����MSE��
    mse = loss_sum / 3
    print("MSE=", mse)

    # �洢��ǰȨ�غͶ�Ӧ�ľ������
    w_list.append(w)
    mse_list.append(mse)

# ��ѡ������Ȩ����MSE�Ĺ�ϵͼ
# plt.plot(w_list, mse_list)
# plt.xlabel('Weight (w)')
# plt.ylabel('Mean Squared Error (MSE)')
# plt.title('Weight vs MSE')
# plt.show()
