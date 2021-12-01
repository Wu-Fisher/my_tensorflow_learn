from keras.datasets import boston_housing
(train_data,train_targets),(test_data,test_targets)=boston_housing.load_data()
# train_data.shape >> (404,13) test_data.shape >> (102,13)

# train_targets >> array([15.2,42.3,50......])

# 面对差距很大的数字，我们采用标准化的手段
# 注意均值和标准差只能从训练集中得到，不能对测试集修改

mean = train_data.mean(axis=0)
train_data -= mean
std=train_data.std(axis=0)
train_data/=std

test_data-=mean
test_data/=std
# 由于样本数据量小，我们使用小网络，防止过拟合

from keras import models
from keras import layers

def build_model():
    model=models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model
# 这里网络最后一层是线性层，如果添加sigmoid，只能预测0～1，这里可以预测任意范围的值
# mse损失函数，均方误差，回归问题常用的损失函数
# 监控新指标MAE 平均绝对误差

# 由于数据量很小，我们使用K折交叉检验 通常取4，5，在K-1个分区上训练，在剩下一个分区上进行评估，验证分数为平均值

import numpy as np

k=4
# //向下取整
num_val_samples = len(train_data)//k
# epoches改为500
num_epochs= 500
#all_scores=[]
#保存没折的验证结果
all_mae_histories=[]

for i in range(k):
    print('processing fold #', i)
    # 准备分区数据
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]

    #准备训练数据：其他分区数据
    #concatenate()拼接数组
    partial_train_data = np.concatenate(
        [train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0
    )

    partial_train_targets =np.concatenate(
        [train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0
    )

    model=build_model()
    # 训练模式（静默模式 verbose = 0）
    history=model.fit(partial_train_data,partial_train_targets,validation_data=(val_data,val_targets),epochs=num_epochs,batch_size=1,verbose=0)
    print(history.history.keys())
    mae_histories=history.history['val_mae']
    all_mae_histories.append(mae_histories)

average_mae_history= [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
print(average_mae_history)
import matplotlib.pyplot as plt

##绘制验证分数，删去前十个差距极大的点
# plt.switch_backend('agg')
def smooth_curve(points,factor=0.9):
    smoothed_points=[]
    for point in points:
        if smoothed_points:
            previous=smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
        return smoothed_points


smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history,'bo')
plt.xlabel('Eco')
plt.ylabel("Va MAE")
plt.show()
plt.savefig("temp_v3.png")

# 80轮后过拟合，最后选择最终版本
exit()



