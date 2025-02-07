import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D

def MQDF_calculater(test, train, h, k):
    x_miu = test-train.mean(axis=0)
    train_cov = np.cov(train, rowvar=False)
    cov_value, cov_vector = np.linalg.eig(train_cov)
    g_2 = 0
    for i in range(k):
        g_2 -= (np.square((cov_vector[i]*x_miu).sum()))*((cov_value[i]-np.square(h))/(cov_value[i]*np.square(h)))
        g_2 += np.log(cov_value[i]*math.pow(h, 2*(4-k)))
    g_2 += np.square((x_miu*x_miu).sum())/np.square(h)
    return g_2





def MQDF(data, n_split, h, k):
    KF = KFold(n_splits=n_split)
    result = []
    for train_index, test_index in KF.split(data):
        train_index, test_index = data.iloc[train_index], data.iloc[test_index]
        train_index, test_index = np.array(train_index), np.array(test_index)

        # fetch the labels of test set
        labels = []
        for samples in test_index:
            labels.append(samples[4])

        # split train dataset with labels
        train_0 = []
        train_1 = []
        train_2 = []
        train_tem = train_index.tolist()
        for samples in train_tem:
            if samples[4] == 0:
                train_0.append(samples[0:4])
            elif samples[4] == 1:
                train_1.append(samples[0:4])
            elif samples[4] == 2:
                train_2.append(samples[0:4])
            else:
                print('There must be something wrong with the dataset!')

        # convert the dataset into np for further calculation
        train_np_0 = np.array(train_0)
        train_np_1 = np.array(train_1)
        train_np_2 = np.array(train_2)

        # calculate accuracy
        accuracy = []
        for sample in test_index:
            score = []
            score.append(MQDF_calculater(sample[0:4], train_np_0, h, k))
            score.append(MQDF_calculater(sample[0:4], train_np_1, h, k))
            score.append(MQDF_calculater(sample[0:4], train_np_2, h, k))
            # print(score)
            accuracy.append(score.index(min(score)))
        accurate = 0
        for i in range(len(accuracy)):
            if accuracy[i] == labels[i]:
                accurate += 1
        result.append(accurate/len(labels))
    # print(result)


    return sum(result)/len(result)




# define each parameters
path = "./data/iris.data"
SEED = 17   # seed for random
n_split = 5      # cross validation
h = 0.001
k = 4

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
# print(train_data)
train_data = np.array(train_data)
train_data = pd.DataFrame(train_data)

score = MQDF(data = train_data, n_split = n_split, h = h, k = k)
print(score)


fig = plt.figure()
ax = plt.axes(projection="3d")

h = np.arange(0.01,2,0.01)
k = np.arange(1,5,1)
x,y = np.meshgrid(h, k)

xx = x.tolist()
yy = y.tolist()
z = []
for i in range(len(xx)):
    zz = []
    for j in range(len(xx[i])):
        zz.append(MQDF(data = train_data, n_split = n_split, h = xx[i][j], k = yy[i][j]))
    print(i)
    z.append(zz)

z = np.array(z)

# z = MQDF(data = train_data, n_split = n_split, h = x, k = y)

ax.plot_surface(x,y,z)
ax.set_xlabel("H Value")
ax.set_ylabel("K Value")
ax.set_zlabel("Accuracy")

ax.set_title("3D surface plot")
plt.show()





