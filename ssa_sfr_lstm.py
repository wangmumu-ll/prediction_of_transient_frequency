# -*- coding: utf-8 -*-
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import random
# import tensorflow.compat.v1 as tf  # tensorflow2.x环境就用这两句
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
from sklearn.preprocessing import MinMaxScaler
from scipy.io import savemat, loadmat
from sklearn.multioutput import MultiOutputRegressor
from sklearn import ensemble
from sklearn import metrics

# In[]

'''
进行适应度计算,以验证集均方差为适应度函数，目的是找到一组超参数 使得网络的误差最小
'''


def fun(pop, P, T, Pt, Tt):
    tf.reset_default_graph()
    tf.set_random_seed(0)
    alpha = pop[0]  # 学习率
    num_epochs = int(pop[1])  # 迭代次数
    hidden_nodes0 = int(pop[2])  # 第一隐含层神经元
    hidden_nodes = int(pop[3])  # 第二隐含层神经元


    input_features = P.shape[1]
    output_class = T.shape[1]
    batch_size = 10  # batchsize
    # placeholder
    X = tf.placeholder("float", [None, input_features])
    Y = tf.placeholder("float", [None, output_class])

    # 定义一个隐层的神经网络
    def RNN(x, hidden_nodes0, hidden_nodes, input_features, output_class):
        x = tf.reshape(x, [-1, 1, input_features])
        # 定义输出层权重
        weights = {'out': tf.Variable(tf.random_normal([hidden_nodes, output_class]))}
        biases = {'out': tf.Variable(tf.random_normal([output_class]))}
        lstm_cell0 = tf.nn.rnn_cell.LSTMCell(hidden_nodes0)
        lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_nodes)
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell0, lstm_cell])
        # 初始化
        init_state = lstm_cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
        outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, initial_state=init_state)

        output_sequence = tf.matmul(tf.reshape(outputs, [-1, hidden_nodes]), weights['out']) + biases['out']
        return tf.reshape(output_sequence, [-1, output_class])

    logits = RNN(X, hidden_nodes0, hidden_nodes, input_features, output_class)
    loss = tf.losses.mean_squared_error(predictions=logits, labels=Y)
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        alpha,
        global_step,
        num_epochs, 0.99,
        staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-10).minimize(loss,
                                                                                            global_step=global_step)
    init = tf.global_variables_initializer()

    # 训练
    with tf.Session() as sess:
        sess.run(init)
        N = P.shape[0]
        for epoch in range(num_epochs):
            total_batch = int(math.ceil(N / batch_size))
            indices = np.arange(N)
            np.random.shuffle(indices)
            avg_loss = 0
            # 迭代训练，顺便计算训练集loss
            for i in range(total_batch):
                rand_index = indices[batch_size * i:batch_size * (i + 1)]
                x = P[rand_index]
                y = T[rand_index]
                _, cost = sess.run([optimizer, loss],
                                   feed_dict={X: x, Y: y})
                avg_loss += cost / total_batch
        # 计算测试集的预测值

        test_pred = sess.run(logits, feed_dict={X: Pt})
        test_pred = test_pred.reshape(-1, output_class)

    F2 = np.mean(np.square((test_pred - Tt)))
    return F2


def boundary(pop, lb, ub):
    # 防止粒子跳出范围,除学习率之外 其他的都是整数
    pop = [int(pop[i]) if i > 0 else pop[i] for i in range(len(lb))]
    for i in range(len(lb)):
        if pop[i] > ub[i] or pop[i] < lb[i]:
            if i == 0:
                pop[i] = (ub[i] - lb[i]) * np.random.rand() + lb[i]
            else:
                pop[i] = np.random.randint(lb[i], ub[i])
    return pop


def SSA(P, T, Pt, Tt):
    M = 10  # 迭代次数
    pop = 10  # 种群数量
    P_percent = 0.2  # 发现者比例
    dim = 4  # 搜索维度,第一个是学习率[0.001 0.01]
    # 第二个是迭代次数[10-100]
    # 第三和第四个是隐含层节点数[1-100]
    Lb = [0.001, 100, 1, 1]
    Ub = [0.02, 200, 100, 100]
    pNum = round(pop * P_percent)  # pNum是生产者
    x = np.zeros((pop, dim))
    fit = np.zeros((pop, 1))
    # 种群初始化
    for i in range(pop):
        for j in range(dim):
            if j == 0:  # 学习率是小数 其他的是整数
                x[i][j] = (Ub[j] - Lb[j]) * np.random.rand() + Lb[j]
            else:
                x[i][j] = np.random.randint(Lb[j], Ub[j])

        fit[i] = fun(x[i, :], P, T, Pt, Tt)
    pFit = fit.copy()
    pX = x.copy()
    fMin = np.min(fit)
    bestI = np.argmin(fit)
    bestX = x[bestI, :].copy()
    Convergence_curve = np.zeros((M,))
    result = np.zeros((M, dim))
    for t in range(M):
        sortIndex = np.argsort(pFit.reshape(-1, )).reshape(-1, )
        fmax = np.max(pFit)
        B = np.argmax(pFit)
        worse = x[B, :].copy()
        r2 = np.random.rand()
        ## 这一部分为发现者（探索者）的位置更新
        if r2 < 0.8:  # %预警值较小，说明没有捕食者出现
            for i in range(pNum):  # r2小于0.8时发现者改变位置
                r1 = np.random.rand()
                x[sortIndex[i], :] = pX[sortIndex[i], :] * np.exp(-i / (r1 * M))
                x[sortIndex[i], :] = boundary(x[sortIndex[i], :], Lb, Ub)
                temp = fun(x[sortIndex[i], :], P, T, Pt, Tt)
                fit[sortIndex[i]] = temp  # 计算新的适应度值
        else:  # 预警值较大，说明有捕食者出现威胁到了种群的安全，需要去其它地方觅食
            for i in range(pNum):  # r2大于0.8时发现者改变位置
                r1 = np.random.rand()
                x[sortIndex[i], :] = pX[sortIndex[i], :] + np.random.normal() * np.ones((1, dim))
                x[sortIndex[i], :] = boundary(x[sortIndex[i], :], Lb, Ub)
                fit[sortIndex[i]] = fun(x[sortIndex[i], :], P, T, Pt, Tt)  # 计算新的适应度值
        bestII = np.argmin(fit)
        bestXX = x[bestII, :].copy()

        ##这一部分为加入者（追随者）的位置更新
        for i in range(pNum + 1, pop):  # 剩下的个体变化
            A = np.floor(np.random.rand(1, dim) * 2) * 2 - 1
            if i > pop / 2:  # 这个代表这部分麻雀处于十分饥饿的状态（因为它们的能量很低，也是是适应度值很差），需要到其它地方觅食
                x[sortIndex[i], :] = np.random.normal() * np.exp((worse - pX[sortIndex[i], :]) / (i ** 2))
            else:  # 这一部分追随者是围绕最好的发现者周围进行觅食，其间也有可能发生食物的争夺，使其自己变成生产者

                x[sortIndex[i], :] = bestXX + np.abs(pX[sortIndex[i], :] - bestXX).dot(
                    A.T * (A * A.T) ** (-1)) * np.ones((1, dim))
            x[sortIndex[i], :] = boundary(x[sortIndex[i], :], Lb, Ub)  # 判断边界是否超出
            fit[sortIndex[i]] = fun(x[sortIndex[i], :], P, T, Pt, Tt)  # 计算适应度值

        # 这一部分为意识到危险（注意这里只是意识到了危险，不代表出现了真正的捕食者）的麻雀的位置更新
        c = random.sample(range(sortIndex.shape[0]),
                          sortIndex.shape[0])  # 这个的作用是在种群中随机产生其位置（也就是这部分的麻雀位置一开始是随机的，意识到危险了要进行位置移动，
        b = sortIndex[np.array(c)[0:round(pop * 0.2)]].reshape(-1, )
        for j in range(b.shape[0]):
            if pFit[sortIndex[b[j]]] > fMin:  # 处于种群外围的麻雀的位置改变
                x[sortIndex[b[j]], :] = bestX + np.random.normal(1, dim) * (np.abs(pX[sortIndex[b[j]], :] - bestX))

            else:  # 处于种群中心的麻雀的位置改变
                x[sortIndex[b[j]], :] = pX[sortIndex[b[j]], :] + (2 * np.random.rand() - 1) * (
                    np.abs(pX[sortIndex[b[j]], :] - worse)) / (pFit[sortIndex[b[j]]] - fmax + 1e-50)
            x[sortIndex[b[j]], :] = boundary(x[sortIndex[b[j]], :], Lb, Ub)
            fit[sortIndex[b[j]]] = fun(x[sortIndex[b[j]], :], P, T, Pt, Tt)  # 计算适应度值

        # 这部分是最终的最优解更新
        for i in range(pop):
            if fit[i] < pFit[i]:
                pFit[i] = fit[i].copy()
                pX[i, :] = x[i, :].copy()

            if pFit[i] < fMin:
                fMin = pFit[i, 0].copy()
                bestX = pX[i, :].copy()
        result[t, :] = bestX
        print(t + 1, fMin, [int(bestX[i]) if i > 0 else bestX[i] for i in range(len(Lb))])

        Convergence_curve[t] = fMin
    return bestX, Convergence_curve, result


# In[] 加载数据
# xlsfile=pd.read_excel('数据集/浙江某地区/bdata1.xls').iloc[0:,1:]# 第一列的日期不作为特征之一
# 数据共97天，每天的数据包括平均温度、最高温度、最低温度、相对湿度、星期类型、与24个时刻的负荷，共29个特征
xlsfile = pd.read_csv('data/sfr_fnp_data_step1.csv').iloc[:, :]
in_1 = xlsfile.drop(xlsfile.columns[len(xlsfile.columns)-1], axis=1)  # axis=1 表示去掉列
in_2 = in_1.drop(in_1.columns[len(in_1.columns)-1], axis=1)
in_ = np.array(in_2[in_2.columns[1:109]])
# 拟用头一天的29个特征与预测当天的前5个特征作为输入，预测当天24个时刻的负荷作为输出
# in_=np.hstack((data[1:,0:5],data[0:-1,:]))
# out_ = [list(t) for t in zip(xlsfile['dfmax_FNP'], xlsfile['tnadir_FNP'])]
out_ = xlsfile[xlsfile.columns[109:111]]
out_ = np.array(out_)
print(in_.shape)#(9999, 8)
print(out_.shape)#(9999, 2)
# out_=data[1:,5:]
# 划分数据，一共96个样本 选择95个样本作为训练集 1个测试集
n = range(in_.shape[0])  # 前95个样本为训练集 最后一天为测试集
m = 6697
train_label = out_[n[0:m],]
test_label = out_[n[m:],]
train_data = in_[n[0:m],]
test_data = in_[n[m:],]
print(train_data.shape)
print(test_data.shape)
print(train_label.shape)
print(test_label.shape)
# 归一化
ss_X = MinMaxScaler(feature_range=(0, 1)).fit(train_data)
ss_y = MinMaxScaler(feature_range=(0, 1)).fit(train_label)
train_data = ss_X.transform(train_data)
train_label = ss_y.transform(train_label)

test_data = ss_X.transform(test_data)
test_label = ss_y.transform(test_label)
best, trace, result = SSA(train_data, train_label, test_data, test_label)
savemat('jieguo/ssa_para.mat', {'trace': trace, 'best': best, 'result': result})
# In[]
trace = loadmat('jieguo/ssa_para.mat')['trace'].reshape(-1, )
result = loadmat('jieguo/ssa_para.mat')['result']
plt.figure()
plt.plot(trace)
plt.title('fitness curve')
plt.xlabel('iteration')
plt.ylabel('fitness value')
plt.savefig("ssa_lstm图片保存/fitness curve.png")

plt.figure()
plt.plot(result[:, 0])
plt.title('learning rate optim')
plt.xlabel('iteration')
plt.ylabel('learning rate value')
plt.savefig("ssa_lstm图片保存/learning rate optim.png")

plt.figure()
plt.plot(result[:, 1])
plt.title('itration optim')
plt.xlabel('iteration')
plt.ylabel('itration value')
plt.savefig("ssa_lstm图片保存/itration optim.png")

plt.figure()
plt.plot(result[:, 2])
plt.title('first hidden nodes optim')
plt.xlabel('iteration')
plt.ylabel('first hidden nodes value')
plt.savefig("ssa_lstm图片保存/first hidden nodes optim.png")

plt.figure()
plt.plot(result[:, 3])
plt.title('second hidden nodes optim')
plt.xlabel('iteration')
plt.ylabel('second hidden nodes value')
plt.savefig("ssa_lstm图片保存/second hidden nodes optim.png")
plt.show()
