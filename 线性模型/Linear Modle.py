import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


w_list = []
mse_list = []

for w in np.arange(0.0, 41, 0.1):
    print("w=", w)
    loss_sum = 0

    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        loss_sum += loss_val
        print("\t", x_val, y_val, y_pred_val, loss_val)

    mse = loss_sum / 3
    print("MSE=", mse)

    w_list.append(w)
    mse_list.append(mse)

# plt.plot(w_list, mse_list)
# plt.xlabel('Weight (w)')
# plt.ylabel('Mean Squared Error (MSE)')
# plt.title('Weight vs MSE')
# plt.show()
