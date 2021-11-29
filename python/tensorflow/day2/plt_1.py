import keras.layers
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
## 1
# digit = train_images[4]
# plt.imshow(digit,cmap=plt.cm.binary)
# plt.show()

## 2
# include head not tail
my_silce =train_images[10:100]
print(my_silce.shape)

## 3 batch
#拆成小批量
batch_one =train_images[:128]

batch_two = train_images[128:256]

#2D (samples,features) 样本，特征  向量数据
#3D (samples,timesteps,features)  ～，时间步长 ，～ （时间）序列
#4D (samples,height,width,channels) 图像数据 ～，～，～，颜色通道
#5D (samples,frames,height,width,channels) 不同视频的一系列帧数

keras.layers.Dense(512,activation='relu')
#该层函数具体如下
#output = relu(dot(W,input)+b)
#dot()点乘，W已知张量，input输入张量,relu(x)=MAX(x,0);

##relu是逐元素运算element-wise

#w和b都是张量，都是该层的属性，被成为权重或可训练参数 kernel 和 bias

#损失函数从梯度方向上减小




