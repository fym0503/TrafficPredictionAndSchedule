import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy.linalg as la
import math
from sklearn.svm import SVR
from process_data import preprocess_data,load_data
import matplotlib.pyplot as plt

def load_data(name):
    traffic = pd.read_csv(name,sep=',',names=['id', 'nounce', 'val1', 'val2'])
    return list(traffic['val2'])#val 1为交通指数 val 2为交通时速
def preprocess_data_svr(data, time_len, rate, seq_len, pre_len):
    
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]
    
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0 : seq_len])
        trainY.append(a[seq_len : seq_len + pre_len])
    for i in range(len(test_data) - seq_len -pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0 : seq_len])
        testY.append(b[seq_len : seq_len + pre_len])
    return trainX, trainY, testX, testY
method='SVR'
data=load_data()
time_len = 10000
num_nodes = 1
train_rate = 0.8
seq_len = 12
pre_len = 1
if method == 'SVR':  
    total_rmse, total_mae, total_acc, result = [],[],[],[]
    length=len(data)
    start_len=100
    a=data
    a_X, a_Y, t_X, t_Y = preprocess_data_svr(a, len(a), train_rate, seq_len, pre_len)
    testY1=[]
    for i in range(start_len,1000):
        
        a = data[:i]
        a_X, a_Y, t_X, t_Y = preprocess_data_svr(a, len(a), train_rate, seq_len, pre_len)
        a_X = np.array(a_X)
        a_X = np.reshape(a_X,[-1, seq_len])
        a_Y = np.array(a_Y)
        a_Y = np.reshape(a_Y,[-1, pre_len])
        a_Y = np.mean(a_Y, axis=1)
        t_X = np.array(t_X)
        t_X = np.reshape(t_X,[-1, seq_len])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y,[-1, pre_len])    
        
        svr_model=SVR(kernel='linear')
        svr_model.fit(a_X, a_Y)
        pre = svr_model.predict(t_X)
        pre = np.array(np.transpose(np.mat(pre)))
        result.append(pre[0])
        testY1.append(t_Y[0])
    result1 = np.array(result)
    result1 = np.reshape(result1, [num_nodes,-1])
    result1 = np.transpose(result1)
    testY1 = np.array(testY1)
    '''
    np.savetxt('1.txt',testY1,delimiter=',',fmt='%s')
    np.savetxt('2.txt',result1,delimiter=',',fmt='%s')
    print(result1)
    '''
    plt.plot(result1[:,0])
    plt.plot(testY1[:,0])
    plt.show()
    testY1 = np.reshape(testY1, [-1,num_nodes])
    total = np.mat(total_acc)
    total[total<0] = 0