import torch
from torch import autograd


def test1():
    x = torch.tensor(2.)
    a = torch.tensor(2., requires_grad=True)  # 表示autograd.grad时，会对其进行求导，
    b = torch.tensor(2., requires_grad=True)
    c = torch.tensor(3., requires_grad=True)

    y = a ** 2 * x + b * x + c

    print('before:', a.grad, b.grad, c.grad)
    grads = autograd.grad(y, [a, b, c])
    print('after', grads[0], grads[1], grads[2])
    print(grads)

    x = torch.tensor([1., 2., 3.], requires_grad=True)
    y = torch.sum(x)

    y.backward()
    print('grad_scaler', x.grad)
    pass

    x = torch.tensor([3., 4.], requires_grad=True)
    if x > torch.tensor([1, 2]):
        y = x
    else:
        y = 0
    y.backward()
    print('x_grad', x.grad)

    # x = torch.tensor([1., 2., 3.], requires_grad=True)
    # y = 2 * x
    # y.backward()
    # print('grad_vector', y.grad)


# 复合函数求导
def test1():
    x = torch.tensor(1., requires_grad=True)  # 表示autograd.grad时，会对其进行求导，
    y = 2 * x + 1
    z = y ** 3

    z.backward()
    print('x.grad =', x.grad)
    print('y.grad =', y.grad)
    print('z.grad =', z.grad)
    pass


# 多元函数求偏导
def test2():
    x = torch.tensor(2.)
    a = torch.tensor(2., requires_grad=True)  # 表示autograd.grad时，会对其进行求导，
    b = torch.tensor(2., requires_grad=True)
    c = torch.tensor(3., requires_grad=True)
    y = a ** 2 * x + b * x + c

    print('before:', a.grad, b.grad, c.grad)
    grads = autograd.grad(y, [a, b, c])
    print('after', grads[0], grads[1], grads[2])
    pass


# 多元函数求偏导
def test3():
    x = torch.tensor([1., 2., 3.], requires_grad=True)
    y = torch.sum(x)
    # y = torch.sum(x * x)
    y.backward()
    print('grad_scaler', x.grad)
    pass


# 分段函数求导 http://www.pointborn.com/article/2021/7/21/1589.html
def test4():
    x = torch.tensor([3., 4.], requires_grad=True)
    if x > torch.tensor([1, 2]):
        y = torch.sum(x)
    else:
        y = 0
    y.backward()
    print('x_grad', x.grad)


# 梯度下降最优化
def test5():
    x = torch.as_tensor([1., -1.])
    x.requires_grad = True
    # f = 5 * x[0] ** 4 + 4 * x[0] ** 2 * x[1] - x[0] * x[1] ** 3 + 4 * x[1] ** 4 - x[0]

    lr = 0.01
    for i in range(10000):
        f = 5 * x[0] ** 4 + 4 * x[0] ** 2 * x[1] - x[0] * x[1] ** 3 + 4 * x[1] ** 4 - x[0]
        f.backward()
        grad = x.grad
        x1 = x + lr * grad
        x = x1
        print(x)
        pass


if __name__ == '__main__':
    # test1()
    # test2()
    # test3()
    # test4()
    test5()
