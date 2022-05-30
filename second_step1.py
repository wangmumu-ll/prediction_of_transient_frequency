# -*- coding: utf-8 -*-
# 两个隐含层的lstm
import numpy as np
# import tensorflow as tf#tensorflow1.x环境就用这个
# import tensorflow.compat.v1 as tf  # tensorflow2.x环境就用这两句
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import savemat
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

tf.reset_default_graph()
tf.set_random_seed(0)
np.random.seed(0)

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

#######################################
#真实数据保存
# save_real_dataframe = pd.DataFrame(test_label)
# save_real_dataframe.to_csv('data/pic_data.csv', index=False, sep=',')
#######################################

train_data = in_[n[0:m],]
test_data = in_[n[m:],]
#######################################
#sfr数据保存
# prev = pd.read_csv('data/pic_data.csv')
# save_sfr_dataframe = pd.DataFrame(test_data[:,-2:])
# new_dataframe = pd.concat([prev,save_sfr_dataframe], axis=1)
# new_dataframe.to_csv('data/pic_data.csv', index=False, sep=',')
#######################################
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

# In[]定义超参数

alpha = 0.001  # 学习率
num_epochs = 100  # 迭代次数
hidden_nodes0 = 64  # 第一隐含层神经元
hidden_nodes = 64  # 第二隐含层神经元
batch_size = 16  # batchsize

# alpha = 0.00318  # 学习率
# num_epochs = 195  # 迭代次数
# hidden_nodes0 = 53  # 第一隐含层神经元
# hidden_nodes = 79  # 第二隐含层神经元
# batch_size = 16  # batchsize
input_features = train_data.shape[1]
output_class = train_label.shape[1]

# placeholder
X = tf.placeholder("float", [None, input_features])
Y = tf.placeholder("float", [None, output_class])


# 定义一个隐层的神经网络
def RNN(x):
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


# In[] 初始化
logits = RNN(X)
loss = tf.losses.mean_squared_error(predictions=logits, labels=Y)
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
    alpha,
    global_step,
    num_epochs, 0.99,
    staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-10).minimize(loss, global_step=global_step)
init = tf.global_variables_initializer()

# In[]训练
train = []
valid = []
with tf.Session() as sess:
    sess.run(init)
    N = train_data.shape[0]
    for epoch in range(num_epochs):
        total_batch = int(math.ceil(N / batch_size))
        indices = np.arange(N)
        np.random.shuffle(indices)
        avg_loss = 0
        # 迭代训练，顺便计算训练集loss
        for i in range(total_batch):
            rand_index = indices[batch_size * i:batch_size * (i + 1)]
            x = train_data[rand_index]
            y = train_label[rand_index]
            _, cost = sess.run([optimizer, loss],
                               feed_dict={X: x, Y: y})
            avg_loss += cost / total_batch

        # 计算测试集loss
        valid_data = test_data
        valid_y = test_label
        valid_loss = sess.run(loss, feed_dict={X: valid_data, Y: valid_y})

        train.append(avg_loss)
        valid.append(valid_loss)
        print('epoch:', epoch, ' ,train loss ', avg_loss, ' ,valid loss: ', valid_loss)

    # 计算训练集与测试集的预测值
    train_pred = sess.run(logits, feed_dict={X: train_data})
    test_pred = sess.run(logits, feed_dict={X: test_data})
# 对测试结果进行反归一化
test_pred = ss_y.inverse_transform(test_pred)
test_label = ss_y.inverse_transform(test_label)
#######################################
#lstm数据保存
prev = pd.read_csv('data/pic_data.csv')
save_lstm_dataframe = pd.DataFrame(test_pred[:,-2:])
new_dataframe = pd.concat([prev,save_lstm_dataframe], axis=1)
new_dataframe.to_csv('data/pic_data.csv', index=False, sep=',')
#######################################
# In[] 画loss曲线
g = plt.figure()
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.plot(train, label='training')
plt.plot(valid, label='testing')
plt.title('loss curve')
plt.ylim(0,0.0002)
plt.legend()

# In[]
# 画出测试集的值

test_pred = np.array(test_pred)
print(test_pred.shape)
n=range(in_.shape[0])
train_data = in_[n[0:m],]
n = range(test_label.shape[0])
# test_label=test_label[n[0:20],].reshape(-1,1)
test_label = test_label[n[0:100],]
n2 = range(test_pred.shape[0])
# test_pred=test_pred[n2[0:20],].reshape(-1,1)
test_pred = test_pred[n2[0:100],]
plt.figure()

# plt.scatter(test_label[:, 0], test_pred[:, 0])
# plt.show()

# plt.plot(test_label,c='r', label='true')
# plt.plot(test_pred,c='b',label='predict')
title = 'LSTM'
plt.title(title)
plt.xlabel('迭代次数')
plt.ylabel('预测结果')
plt.legend()
# plt.show()
x = range(1, 101)
plt.scatter(x, test_label[:, 0], marker='_', c='k', s=40, label='真实最低频率')
# plt.scatter(x,test_label[:,0], c='r')
plt.scatter(x, test_pred[:, 0], marker='|', c='k', s=20, label='预测最低频率')
plt.scatter(x, test_label[:, 1], marker='o', c='none',edgecolors='k', s=40, label='真实最低频率时刻')
# plt.scatter(x,test_label[:,0], c='r')
plt.scatter(x, test_pred[:, 1], marker='x', c='k', s=10, label='预测最低频率时刻')
# plt.scatter(x, sometest[:,0], c='r', label='true dfmax')
# plt.scatter(x,sometest[:,1], c='b', label='true tnadir')
# plt.scatter(x,somepredict[:,0], c='m', label='predict dfmax')
# plt.scatter(x,somepredict[:,1], c='c', label='predict tnadir')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.gcf().subplots_adjust(left=None,top=None,bottom=None, right=0.7)
plt.rcParams['figure.figsize'] = (12.0, 8.0) # 单位是inches
plt.savefig("ssa_lstm图片保存/LSTM.png")
plt.show()

#------------------------------------------------------------------------------------
# somepredict = test_pred[:1000]
# sometest = test_label[:1000]
# print(somepredict.shape)
# print(sometest.shape)
# x = range(1,1001)
# # print(x)
# # print(x.shape, sometest[:,0].shape, sometest[:,1].shape)
# plt.scatter(x, sometest[:,0], c='r', label='true dfmax')
# plt.scatter(x,sometest[:,1], c='b', label='true tnadir')
# plt.scatter(x,somepredict[:,0], c='m', label='predict dfmax')
# plt.scatter(x,somepredict[:,1], c='c', label='predict tnadir')
# # plt.ylim([-4,9])
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
# # plt.legend()
# plt.gcf().subplots_adjust(left=None,top=None,bottom=None, right=0.7)
# plt.rcParams['figure.figsize'] = (12.0, 8.0) # 单位是inches
# plt.show()
#------------------------------------------------------------------------------------


# savemat('结果/lstm_result.mat', {'true': test_label, 'pred': test_pred})
# In[]计算各种指标
# mape
test_mape = np.mean(np.abs((test_pred - test_label) / test_label))
# rmse
test_rmse = np.sqrt(np.mean(np.square(test_pred - test_label)))
# mae
test_mae = np.mean(np.abs(test_pred - test_label))
mae_ = metrics.mean_absolute_error(test_pred, test_label)
print('testmae')
print(test_mae)
print(mae_)
# mse
test_mse = np.mean(np.square(test_pred - test_label))
# R2
test_r2 = r2_score(test_label, test_pred)
# TIC
test_tic = np.sqrt(np.mean(np.square(test_pred - test_label))) / (
            np.sqrt(np.mean(np.square(test_label))) + np.sqrt(np.mean(np.square(test_pred))))

print('LSTM测试集的mape:', test_mape, ' rmse:', test_rmse, ' mae:', test_mae, "mse； ", test_mse, ' R2:', test_r2, ' tic:',
      test_tic)