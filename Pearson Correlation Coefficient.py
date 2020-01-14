import numpy as np
import pandas as pd
import math
import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

# Pre-processing
raw_data = pd.read_csv('veh-prime.csv')
df = np.array(raw_data)
data = pd.DataFrame(raw_data)
data.loc[data['CLASS'] == 'car', 'CLASS'] = 1
data.loc[data['CLASS'] == 'noncar', 'CLASS'] = 0

'''
loc（） choose specific column through label， 
The second parameter can confine the column you want to process instead of the whole row.
'''
data_f = data
temp1 = preprocessing.scale(data_f.iloc[:, :-1])
temp2 = data_f.iloc[:, -1]
data_f = np.column_stack((temp1, temp2))
data_f = pd.DataFrame(data_f)
data = np.array(data)


#  Method of computation
def pearson(x, y):
    sum_sq_x = 0
    sum_sq_y = 0
    sum_coproduct = 0
    mean_x = 0
    mean_y = 0
    N = len(x)
    for i in range(0, N):
        sum_sq_x += x[i] * x[i]
        sum_sq_y += y[i] * y[i]
        sum_coproduct += x[i] * y[i]
        mean_x += x[i]
        mean_y += y[i]
    mean_x = mean_x / N
    mean_y = mean_y / N
    pop_sd_x = math.sqrt((sum_sq_x / N) - (mean_x * mean_x))
    pop_sd_y = math.sqrt((sum_sq_y / N) - (mean_y * mean_y))
    cov_x_y = (sum_coproduct / N) - (mean_x * mean_y)
    correlation = cov_x_y / (pop_sd_x * pop_sd_y)
    return correlation  # get|r| instead of r


feature = []
rank = []
for i in range(0, data.shape[1] - 1):
    # Pearson_Index.append(pearson(data1[:, i], data1[:, -1]))
    # print('R(f', (i), ') = ', pearson(data[:, i], data[:, -1]))
    feature.append([i, np.abs(pearson(data[:, i], data[:, -1]))])
    rank = sorted(feature, key=operator.itemgetter(1), reverse=True)
print('Feature Selection:')
for item in rank:
    print('f', item[0], end=' ', sep='')
print('')
Pearson_Index = []
for item in rank:
    Pearson_Index.append(item[0])
'''Question 1 (a)
for i in range(0, data.shape[1] - 1):
    # Pearson_Index.append(pearson(data1[:, i], data1[:, -1]))
    print('R(f', (i), ') = ', pearson(data[:, i], data[:, -1]))
'''


def KNN(Train_X, Train_Y, Test_X, Test_Y):
    KNN = KNeighborsClassifier(n_neighbors=7)  # Define a KNN Classifier
    # KNN.fit(data[:10, 0:-1], data[:10, -1])  # If the sample is less than the min request of KNN, will get error
    # Prediction = KNN.predict(data[-10:len(data), 0:-1])
    # accuracy = KNN.score(data[-10:len(data), 0:-1], data[-10:len(data), -1])
    KNN.fit(Train_X, Train_Y)  # If the sample is less than the min request of KNN, will get error
    # Prediction = KNN.predict(Test_X)
    accuracy = KNN.score(Test_X, Test_Y)
    # print(Prediction, accuracy)
    return accuracy


def LOOCV_KNN(data):
    fold = len(data)  # Well.. since it is LOOCV, which means the K = N, let's do some modification on my CV
    # for index in Pearson_Index:
    # if index!=index[0]:
    # np.column_stack((data1[:,4], data1[:, index]))
    # data1[]
    SumAccuracy = []
    for j in range(0, len(data), int(len(data) / fold)):
        Train_set = np.row_stack(
            (data[0:j], data[(j + int(len(data) / fold)):len(data)]))  # Select Train set (except the jth fold)
        Test_set = data[j:j + int(len(data) / fold)]  # select the jth fold
        Train_X = Train_set[:, 0:-1]
        Train_Y = Train_set[:, -1]
        Test_X = Test_set[:, 0:-1]
        Test_Y = Test_set[:, -1]
        CAccur = KNN(Train_X, Train_Y, Test_X, Test_Y)
        SumAccuracy.append(CAccur)
    Accuracy = np.mean(SumAccuracy)
    Accuracy = np.round(Accuracy * 100, 2)
    print('Accuracy: ''%.2f' % Accuracy, '%')
    print('')
    return Accuracy


Array_Index = []
Available_Index = []  # Index of features we need for each test
for i in range(1, len(Pearson_Index) + 1):
    for j in range(0, i):
        Available_Index.append(Pearson_Index[j])
    Array_Index.append(Available_Index)
    Available_Index = []
print(Array_Index)
for i in Array_Index:
    print(i)
    print(len(i))
# x=[2,7,8,10,14,19,20]
m = 1
rank = []
for sample in Array_Index:
    # print(len(data))
    # print(np.r_[Pearson_Index, -1])  # row connect
    # data2 = np.column_stack((data[:,4],data[:,13]))
    data3 = data_f.iloc[:, np.r_[sample, -1]]  # learn some about the lic and lioc, you will find some benefits from it,
    # which should be the basic knowledge you need to keep in mine.
    # print(data3)
    data3 = np.array(data3)
    # x = input()
    # data2 = np.column_stack((data3[:, 0:-1], data3[:, -1]))
    # data = data[:,0:data.shape[1]]
    print('When m = ', m, end=' ')
    rank.append(LOOCV_KNN(data3))
    m += 1
print('The Optical m is 20 in this case')
