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
num_epochs= 100
all_scores=[]

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
    model.fit(partial_train_data,partial_train_targets,epochs=num_epochs,batch_size=1,verbose=0)
    val_mse ,val_mae = model.evaluate(val_data,val_targets,verbose=0)
    all_scores.append(val_mae)

print("all_scores:",all_scores)
print("mean",np.mean(all_scores))

exit()



