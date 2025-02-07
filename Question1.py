import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def P_x_wk(train_index, x, h, nk, classification):
    sum = 0
    for samples in train_index:
        if samples[4] == classification:
            u = (x - samples[0:4])/h
            parameter = 1/np.sqrt(2*math.pi)
            fei = parameter * math.exp(-(np.square(u[0])+np.square(u[1])+np.square(u[2])+np.square(u[3]))/2)
            sum += fei/(math.pow(h, 4))
    pxwk = sum/nk
    return pxwk

def parzen_window_method(data, h, n_split):
    KF = KFold(n_splits=n_split)
    parzen = []
    for train_index, test_index in KF.split(data):
        train_index, test_index = data.iloc[train_index], data.iloc[test_index]
        train_index, test_index = np.array(train_index), np.array(test_index)

        # fetch the labels of test set
        labels = []
        for samples in test_index:
            labels.append(samples[4])

        # calculate for P_wk of train set
        P_wk = np.array([0, 0, 0])
        for sample in train_index:
            if sample[4] == 0:
                P_wk[0] += 1
            elif sample[4] == 1:
                P_wk[1] += 1
            elif sample[4] == 2:
                P_wk[2] += 1
            else:
                print('There is something wrong with the labels!')
        num = P_wk[0] + P_wk[1] + P_wk[2]
        P_wk_pro = P_wk/num

        # calculate for scores
        predict = []
        for samples in test_index:
            scores = []
            for i in range(3):
                P_wk_x = P_wk_pro[i] * P_x_wk(train_index=train_index, x=samples[0:4], h=h, nk=P_wk[i],
                                          classification=i)
                scores.append(P_wk_x)
            predict.append(scores.index(max(scores)))
        # predict compares with labels
        count = 0
        for i in range(len(labels)):
            if predict[i] == labels[i]:
                count += 1
        result = count/len(labels)

        # propabilities of each epoch
        parzen.append(result)
        # return max(parzen)
    # print(parzen)



    return sum(parzen)/len(parzen)

# define each parameters
path = "./data/iris.data"
SEED = 17   # seed for random
n_split = 5      # cross validation
h = 0.001

# load training data
df = pd.read_csv(path, header=None, sep=',')
data = np.array(df)
train_data = data.tolist()
# print(list_test)
# print(type(list_test[0][4]))

# translate targets into 0 1 2
for list in train_data:
    if list[4]=='Iris-setosa':
        list[4] = 0
    elif list[4]=='Iris-versicolor':
        list[4] = 1
    elif list[4]=='Iris-virginica':
        list[4] = 2
    else:
        print('There are some mistakes in the train_data!')


# random the data with SEED is 17
random.seed(SEED)
random.shuffle(train_data)
print(train_data)
train_data = np.array(train_data)
train_data = pd.DataFrame(train_data)

# h_line = np.linspace(0.002, 18, 17999)
# lineline = []
# indeex = 0
# for draw in h_line:
#     print(indeex)
#     score = parzen_window_method(data = train_data, h = draw, n_split = n_split)
#     lineline.append(score)
#     indeex += 1
# np.save('y_modified.npy', lineline)
#
# plt.figure()
# plt.plot(h_line, lineline)
# plt.show()

# calculate the accuracy of parzen window method
score = parzen_window_method(data = train_data, h = h, n_split = n_split)
print(score)


