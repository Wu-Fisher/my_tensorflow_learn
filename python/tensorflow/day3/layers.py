from keras import layers
from keras import models
# 创建一个直接收第一个维度大小为784的2D张量，（0轴是批量维度，其大小没有指定，因此可以取任意值）
# 该层返回的张量会变成32
# layer = layers.Dense(32,input_shape=(784,))

# print(layer)

# model= models.Sequential()
# model.add(layers.Dense(32,input_shape=(784,)))
# model.add(layers.Dense(32))

# 二分类问题 二元交叉熵（binary crossentropy）
# 多分类问题 分类交叉熵（categorical crossentropy）
# 回归问题 均方误差（mean-squared err）
# ...

#定义模型的两种方法
# Sequential类 层的线性堆叠

# model= models.Sequential()
# model.add(layers.Dense(32,activation='relu',input_shape=(784,)))
# model.add(layers.Dense(10,actibation='softmax'))

# functional API 层组成的有向无环图

input_tensor = layers.Input(shape=(784,))
x=layers.Dense(32,activation='relu')(input_tensor)
output_tensor=layers.Dense(10,activation='softmax')(x)
model =models.Model(inputs=input_tensor,outputs=output_tensor)
## 括号之间相当与联系起来了，虽然写在了不同的表达式里面

##

from keras import optimizers
#
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='mse',metrics=['accuracy'])
# model.fit(input_tensor,target_tensor,batch_size=128,epochs=10)


# face the version question of optimizer
# from tensorflow.python.keras.optimizers import adam_v2
# from tensorflow.python.keras.optimizers import rmsprop_v2
# optimizer =adam_v2.Adam(lr=1e-3)