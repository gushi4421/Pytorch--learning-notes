x_data=[1.0,2.0,3.0]
y_data=[2.0,5.0,6.0]

w=1.0
a=0.01  #学习率
def forward(x):
    return x*w

def cost(xs,ys):
    cost=0
    for x,y in zip(xs,ys)
        y_pred=forward(x)
        cost+=(y_pred-y)**2
    return cost/len(xs)

#计算梯度
def gradient(xs,ys):
    grad=0
    for x,y in zip(xs,ys):
        grad+=2*x*(x*w-y)
    return grad/len(xs)

print("Prediect (before training)", 4, forward(4))

for epoch in range(1,101):
    cost_val=cost(x_data,y_data)
    grad_val=gradient(x_data,y_data)
    w-=