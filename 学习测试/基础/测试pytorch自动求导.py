#coding=gb2312

""" 2022.10.9
"""
import torch
from torch import autograd


def test_������������():
    x = torch.tensor(2.)  # ����ǰ��ĵ��ʾʡ��0
    a = torch.tensor(2., requires_grad=True)  # ��ʾautograd.gradʱ�����������󵼣�
    b = torch.tensor(2., requires_grad=True)
    c = torch.tensor(3., requires_grad=True)

    print("1������autograd.grad����������")
    y = a ** 2 * x + b * x + c  # **��ʾ������

    """ �ݶ��½��㷨���̣�
    1��������ֵ
    2�������ֵ�ĵ���
    3��x1 = x0 + lr*d �õ��µĳ�ֵ������lrΪ������dΪ�ݶȡ�
    4��ѭ������2��3��,ֱ��x1-x0��С����õ�y��Сֵ��Ӧ��x
    �����y��Ϊloss,�������ݶ��½��㷨���õ�ʹy��Сʱ�ľ���ˡ�����������ȵȣ�������ģΪ��ʮ�򼶱�
    """

    # �ݶ��ֶ��Ƶ�
    # dy/da = 2xa = 8   (����x=2,a=2)
    # dy/db = x = 2     (����x=2)
    # dy/dc = 1
    print('before:', a.grad, b.grad, c.grad)
    grads = autograd.grad(y, [a, b, c])
    print('after', grads[0], grads[1], grads[2])
    pass

    print("2��������������ķ���backward������")
    a.grad = None
    b.grad = None
    c.grad = None
    y = a ** 2 * x + b * x + c  # **��ʾ������
    y.backward()
    print('backward', a.grad, b.grad, c.grad)

    # �����������󵼣�y = x1+x2+x3, dy/dx1 = 1
    x = torch.tensor([1., 2., 3.], requires_grad=True)
    y = torch.sum(x)
    y.backward()
    print(f'�����������󵼣��ֶ��Ƶ����Ϊ��[dy/dx1 = 1, dy/dx2 = 1, dy/dx3 = 1], pytorch������Ϊ�� {x.grad}')
    pass


# ���Ϻ�����
def test_���Ϻ�����():
    x = torch.tensor(1., requires_grad=True)  # ��ʾautograd.gradʱ�����������󵼣�
    y = 2 * x + 1
    z = y ** 3

    # ���Ϻ�����,�ֶ��Ƶ����Ϊ��dz/dx = dz/dy * dy/dx = 3*y^2 * 2 = 3*(2*x+1)^2*2 = 54
    z.backward()
    print(f'���Ϻ�����,�ֶ��Ƶ����Ϊ��dz/dx = dz/dy * dy/dx = 3*y^2 * 2 = 3*(2*x+1)^2*2 = 54, pytorch������Ϊ��x.grad = {x.grad}')
    print('y.grad =', y.grad)
    print('z.grad =', z.grad)
    pass


# �ֶκ����� http://www.pointborn.com/article/2021/7/21/1589.html
def test_�ֶκ�����():
    # x = torch.tensor([3., 4.], requires_grad=True)
    # if x > torch.tensor([1, 2]):
    #     y = torch.sum(x)
    # else:
    #     y = 0
    # y.backward()
    # print('x_grad', x.grad)
    pass


# �ݶ��½��㷨��Сֵ����Ŀ����������ֵ�������������룬2014����13.6��������һ����
def test_�ݶ��½�������Сֵ():
    x = torch.tensor([1., -1.], requires_grad=True)
    lr = 0.1
    for i in range(200):
        f = 5 * x[0] ** 4 + 4 * x[0] ** 2 * x[1] - x[0] * x[1] ** 3 + 4 * x[1] ** 4 - x[0]
        # f.backward() ����Ҷ�ڵ����ۻ��ݶȡ�out����Ҷ�ڵ㣬���gradΪNone��autograd.grad�����ڲ����κ�����wrt���κ��������ݶȡ�
        grad = autograd.grad(f, x)
        grad = grad[0].data
        x1 = x - lr * grad
        if (x1 - x).norm()<1E-5:
            break
        x = x1
        #x.grad = None
        print(x)
        pass


if __name__ == '__main__':
    test_������������()
    test_���Ϻ�����()
    test_�ֶκ�����()
    test_�ݶ��½�������Сֵ()
