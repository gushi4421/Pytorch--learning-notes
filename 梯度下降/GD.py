x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 5.0, 6.0]

w = 1.0
rate = 0.01


def forward(x):
    return x * w


def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


print("Predict (before training)", 4, forward(4))

for epoch in range(1, 101):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= rate * grad_val
    print("Epoch:", epoch, "w=", w, "cost=", cost_val)

print("Predict (after training)", 4, forward(4))
