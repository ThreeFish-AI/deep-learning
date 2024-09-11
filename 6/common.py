import numpy as np


######################## 激活函数 ########################


def sigmoid(x):
    """S 型函数"""
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """归一化指数函数"""
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


######################## 损失函数 ########################


def cross_entropy_error(y, t):
    """交叉熵误差函数"""
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是 one-hot-vector 的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


######################## 梯度函数 ########################
#                        数值微分法                      #


def _numerical_gradient_1d(f, x):
    """梯度函数
    用数值微分求导法，求 f 关于 1 组参数的 1 个梯度。

    Args:
        f: 损失函数
        x: 参数（1 组参数，1 维数组）
    Returns:
        grad: 1 组梯度（1 维数组）
    """
    h = 1e-4                    # 0.0001
    grad = np.zeros_like(x)     # 生成和 x 形状相同的数组，用于存放梯度（所有变量的偏导数）

    for idx in range(x.size):   # 挨个遍历所有变量
        xi = x[idx]             # 取第 idx 个变量
        x[idx] = float(xi) + h
        fxh1 = f(x)             # 求第 idx 个变量增大 h 所得计算结果

        x[idx] = xi - h
        fxh2 = f(x)             # 求第 idx 个变量减小 h 所得计算结果

        grad[idx] = (fxh1 - fxh2) / (2*h)  # 求第 idx 个变量的偏导数
        x[idx] = xi             # 还原第 idx 个变量的值
    return grad


def numerical_gradient_2d(f, X):
    """梯度函数（批量）
    用数值微分求导法，求 f 关于 n 组参数的 n 个梯度。

    Args:
        f: 损失函数
        X: 参数（n 组参数，2 维数组）
    Returns:
        grad: n 组梯度（2 维数组）
    """
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)

        return grad


######################## 梯度函数 ########################
#                        反向传播算法                    #

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


######################## mini batch #####################

def mini_batch(x, t, batch_size=100):
    """从 (x, t) 随机选取 mini-batch 数据

    Args:
        x: 数据集
        t: 标签集
        batch_size: mini-batch 大小
    Returns:
        x_batch: mini-batch 数据
        t_batch: mini-batch 标签
    """

    train_size = x.shape[0]
    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x[batch_mask]
    t_batch = t[batch_mask]

    return x_batch, t_batch


######################## one hot #####################

def one_hot_label(X):
    """将标签数据批量转换为长度为 10 的独热数组"""
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T
