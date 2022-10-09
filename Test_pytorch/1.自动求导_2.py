import torch


def test1():
    x = torch.tensor(1., requires_grad=True)  # 表示autograd.grad时，会对其进行求导，
    y = 2 * x + 1
    z = y ** 3
    # t = torch.tensor(3.)

    z.backward()
    print('x.grad =', x.grad)
    print('y.grad =', y.grad)
    print('z.grad =', z.grad)
    pass


def test2():
    x = torch.tensor(2., requires_grad=True)
    y = x * x * x
    z = y + y
    pass


if __name__ == '__main__':
    test1()
    # test2()
