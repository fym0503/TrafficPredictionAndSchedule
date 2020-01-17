# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy.linalg as la
import math
from sklearn.svm import SVR
from process_data import preprocess_data,load_data
import matplotlib.pyplot as plt
###### evaluation ######
def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b)/la.norm(a)
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a - b))/np.var(a)
    return rmse, mae, 1-F_norm, r2, var

time_len = 10000
num_nodes = 1
train_rate = 0.8
seq_len = 12
pre_len = 3
trainX,trainY,testX,testY = preprocess_data(time_len, train_rate, seq_len, pre_len)
data=load_data()
'''
print(len(trainX))
print(len(trainY))
print(len(testX))
print(len(testY))
'''
method = 'HA' ####HA or SVR or ARIMA

########### HA #############
if method == 'HA':
    result = []
    for i in range(len(testX)):
        a = testX[i]
        a1 = np.mean(a, axis=0)
        a = testX[i]
        a1 = np.mean(a, axis=0) 
        result.append(a1)
        
    result1 = np.array(result)
    result1 = np.reshape(result1, [-1,num_nodes])
    testY1 = np.array(testY[:,0])
    testY1 = np.reshape(testY1, [-1,num_nodes])
    rmse, mae, accuracy,r2,var = evaluation(testY1, result1)  
    plt.plot(testY1)
    plt.title("History average")
    plt.plot(result1)
    plt.show()
    print('HA_rmse:%r'%rmse,
          'HA_mae:%r'%mae,
          'HA_acc:%r'%accuracy,
          'HA_r2:%r'%r2,
          'HA_var:%r'%var)



######## ARIMA #########
'''
if method == 'ARIMA':
    rng = pd.date_range('1/3/2012', periods=5664, freq='15min')
    a1 = pd.DatetimeIndex(rng)
    data.index = a1
    num = data.shape[1]   
    rmse,mae,acc,r2,var,pred,ori = [],[],[],[],[],[],[]
    for i in range(156):
        ts = data.iloc[:,i]
        ts_log=np.log(ts)    
        ts_log=np.array(ts_log,dtype=np.float)
        where_are_inf = np.isinf(ts_log)
        ts_log[where_are_inf] = 0
        ts_log = pd.Series(ts_log)
        ts_log.index = a1
        model = ARIMA(ts_log,order=[1,0,0])
        properModel = model.fit()
        predict_ts = properModel.predict(4, dynamic=True)
        log_recover = np.exp(predict_ts)
        ts = ts[log_recover.index]
        er_rmse,er_mae,er_acc,r2_score,var_score = evaluation(ts,log_recover)
        rmse.append(er_rmse)
        mae.append(er_mae)
        acc.append(er_acc)
        r2.append(r2_score)
        var.append(var_score)
#    for i in range(109,num):
#        ts = data.iloc[:,i]
#        ts_log=np.log(ts)    
#        ts_log=np.array(ts_log,dtype=np.float)
#        where_are_inf = np.isinf(ts_log)
#        ts_log[where_are_inf] = 0
#        ts_log = pd.Series(ts_log)
#        ts_log.index = a1
#        model = ARIMA(ts_log,order=[1,1,1])
#        properModel = model.fit(disp=-1, method='css')
#        predict_ts = properModel.predict(2, dynamic=True)
#        log_recover = np.exp(predict_ts)
#        ts = ts[log_recover.index]
#        er_rmse,er_mae,er_acc,r2_score,var_score = evaluation(ts,log_recover)
#        rmse.append(er_rmse)
#        mae.append(er_mae)
#        acc.append(er_acc)  
#        r2.append(r2_score)
#        var.append(var_score)
    acc1 = np.mat(acc)
    acc1[acc1 < 0] = 0
    print('arima_rmse:%r'%(np.mean(rmse)),
          'arima_mae:%r'%(np.mean(mae)),
          'arima_acc:%r'%(np.mean(acc1)),
          'arima_r2:%r'%(np.mean(r2)),
          'arima_var:%r'%(np.mean(var)))
  
'''