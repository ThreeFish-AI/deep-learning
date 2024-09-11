from datetime import datetime

from sklearn.datasets import fetch_openml

from common import mini_batch
from simple_net import SimpleNet

mnist = fetch_openml('mnist_784', version=1)
# "data" 是图像数据，"target" 是标签数据
X, y = mnist["data"], mnist["target"]
# 此处直接取前 60000 个样例为训练集，后 10000 个样例为测试集
x_train, x_test, t_train, t_test = X[:60000], X[60000:], y[:60000], y[60000:]

iters_num = 10000           # 设定迭代次数：让 SimpleNet 对训练集进行 10000 次学习，每次学习随机选取 100 个样例
batch_size = 100            # 设定 mini-batch 的大小：每次从训练集中随机选取 100 个样例进行学习
learning_rate = 0.1         # 设定学习率：每次学习时更新权重参数的步长为 0.1

train_loss_list = []        # 记录训练过程中的损失值
train_acc_list = []         # 记录每轮学习后，神经网络在训练集上的识别精度
test_acc_list = []          # 记录每轮学习后，神经网络在测试集上的识别精度

# 每轮学习的迭代次数 = 训练集长度 / 每批长度
train_size = x_train.shape[0]
iter_per_epoch = max(train_size / batch_size, 1)

network = SimpleNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    print(f"Iteration {i}, {datetime.now()}")
    # 执行步骤 1
    x_batch, t_batch = mini_batch(x_train, t_train, batch_size)
    grad = network.numerical_gradient(x_batch, t_batch)
    print(f"Iteration {i}, step 1, {datetime.now()}")

    # 执行步骤 2
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    print(f"Iteration {i}, step 2, {datetime.now()}")

    # 记录训练过程中的损失值
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print(f"Iteration {i}, step 3, {datetime.now()}")

    # 每轮（600 次）学习记录一次训练数据和测试数据的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"train acc, test acc | {str(train_acc)}, {str(test_acc)}")
