import numpy as np
import pandas as pd
import math
import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import time
# Pre-processing
#time = time.clock()
#print('1')
#time = time.clock()
#print(time)
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

feature=[]
rank=[]
for i in range(0, data.shape[1] - 1):
    # Pearson_Index.append(pearson(data1[:, i], data1[:, -1]))
    #print('R(f', (i), ') = ', pearson(data[:, i], data[:, -1]))
    feature.append([i,np.abs(pearson(data[:, i], data[:, -1]))])
    rank = sorted(feature,key=operator.itemgetter(1),reverse=True)
print('Feature Selection:')
for item in rank:
    print('f',item[0],end=' ',sep='')
print('')
Pearson_Index = []
for item in rank:
    Pearson_Index.append(item[0])
'''Question 1 (a)
for i in range(0, data.shape[1] - 1):
    # Pearson_Index.append(pearson(data1[:, i], data1[:, -1]))
    print('R(f', (i), ') = ', pearson(data[:, i], data[:, -1]))
'''
#  Getting Sorted Data From weka
#Pearson_Index = [5, 14, 15, 17, 8, 23, 27, 2, 21, 32, 35, 3, 29, 26, 20, 18, 33, 9, 1, 11, 22, 12, 34, 7, 16, 36, 30,
                 #19, 28, 10, 4, 31, 25, 24, 13, 6]
#for i in range(0, len(Pearson_Index)):
    #Pearson_Index[i] -= 1
#print(Pearson_Index)


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
    Accuracy = np.round(Accuracy*100,2)
    #print('Accuracy: ''%.2f' % Accuracy, '%')
    #print('')
    return Accuracy

Array_Index = []
Available_Index = []  # Index of features we need for each test
for i in range(1, len(Pearson_Index)+1):
    for j in range(0, i):
        Available_Index.append(Pearson_Index[j])
    Array_Index.append(Available_Index)
    Available_Index = []
#print(Array_Index)
#for i in Array_Index:
    #print(i)
    #print(len(i))
'''
#x=input()
# The usage of KNN of sk-learn
Sum=[]
set1=[i for i in range(0,36)]
chosen = []
set2= []
for i in set1:
    data4 = data_f.iloc[:, np.r_[i, -1]]
    data4 = np.array(data4)
    #print('When choice = ', i, end=' ')
    Sum.append([i,LOOCV_KNN(data4)])
Sum1 = sorted(Sum,key=operator.itemgetter(1), reverse=True)
#set1.remove(Sum1[0][0])
set2.append(Sum1[0][0])
#print(Sum1[0][0],Sum1[0][1])
chosen.append([Sum1[0][0],Sum1[0][1]])
print(chosen.pop())
#print('When choice = ',chosen[0],'accuracy is')
'''
Sum=[]
set1=[i for i in range(0,36)]
chosen = []
set2= []
flag=-1
current_accuracy=0

while(current_accuracy>flag):
    flag=current_accuracy
    for i in set1:
        set2.append(i)
        data4 = data_f.iloc[:, np.r_[set2, -1]]
        data4 = np.array(data4)
        # print('When choice = ', i, end=' ')
        Sum.append([i, LOOCV_KNN(data4)])  # 记录每次遍历得到的所有accuracy和对应index
        set2.pop()
    Sum1 = sorted(Sum, key=operator.itemgetter(1), reverse=True)
    set1.remove(Sum1[0][0])
    set2.append(Sum1[0][0])
    current_accuracy = Sum1[0][1]
    if(current_accuracy>flag):
        print('Chosen Subset: ',set2,'  ',current_accuracy,'%')
    Sum = []

