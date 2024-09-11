from common import *

import numpy as np


class SimpleNet(object):
    """一个简单的演示神经网络 SimpleNet，用于演示神经网络对手写数字图像识别任务的自动学习和推理过程。

    Attributes:
        params: 存放 SimpleNet 网络权重参数与偏置参数
            W1: 第 1 层网络的权重参数
            b1: 第 1 层网络的偏置参数
            W2: 第 2 层网络的权重参数
            b2: 第 2 层网络的偏置参数
    """

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """SimpleNet 的初始化函数

        Args:
            input_size:      输入层（第 0 层）神经元个数（神经网络入参个数）
            hidden_size:     隐藏层（第 1 层）神经元个数
            output_size:     输出层（第 2 层）神经元个数（神经网络出参个数）
            weight_init_std: 用于初始化权重参数的高斯分布的标准差
        """
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)    # 用高斯分布进行 W1 参数的随机初始化
        self.params['b1'] = np.zeros(hidden_size)       # 用 0 进行 b1 参数的初始化
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)   # 用高斯分布进行 W2 参数的随机初始化
        self.params['b2'] = np.zeros(output_size)       # 用 0 进行 b2 参数的初始化

    def predict(self, x):
        """推理函数
            识别数字图像代表的数值。

        Args:
            x: 图像像素值数组（图像数据）
        Returns:
            y: 推理结果，图像代表的数值
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        s1 = np.dot(x, W1) + b1
        a1 = sigmoid(s1)
        s2 = np.dot(a1, W2) + b2
        y = softmax(s2)

        return y

    def loss(self, x, t):
        """损失函数（交叉熵误差）

        Args:
            x: 输入数据，即图像数据
            t: 监督数据，即正确解标签
        Returns:
            loss: 推理的损失值
        """
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def numerical_gradient(self, x, t):
        """梯度函数（数值微分求导法）

        Args:
            x: 输入数据，即图像数据
            t: 监督数据，即正确解标签
        Returns:
            grads: 误差函数关于当前权重参数的梯度
        """
        def loss_W(W):
            """损失值关于权重参数的函数
            """
            return self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient_2d(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient_2d(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient_2d(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient_2d(loss_W, self.params['b2'])

        return grads

    def accuracy(self, x, t):
        """精准度函数
        求推理正确的百分比。
        Args:
            x: 输入数据，即图像数据
            t: 监督数据，即正确解标签
        Returns:
            accuracy: 推理的精准度
        """

        y = self.predict(x)
        y = np.argmax(y, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
