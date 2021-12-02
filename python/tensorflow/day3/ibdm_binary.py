from keras.datasets import imdb
(train_data,train_labels),(test_data,test_labels)= imdb.load_data(num_words=10000)

#print(train_data)
##(25000,)代表二维长度不均

# print( max([max(sequence) for sequence in train_data]))
#  [list([1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32])
#  list([1, 194, 1153, 194, 8255, 78, 228, 5, 6, 1463, 4369, 5012, 134, 26, 4, 715, 8, 118, 1634, 14, 394, 20, 13, 119, 954, 189, 102, 5, 207, 110, 3103, 21, 14, 69, 188, 8, 30, 23, 7, 4, 249, 126, 93, 4, 114, 9, 2300, 1523, 5, 647, 4, 116, 9, 35, 8163, 4, 229, 9, 340, 1322, 4, 118, 9, 4, 130, 4901, 19, 4, 1002, 5, 89, 29, 952, 46, 37, 4, 455, 9, 45, 43, 38, 1543, 1905, 398, 4, 1649, 26, 6853, 5, 163, 11, 3215, 2, 4, 1153, 9, 194, 775, 7, 8255, 2, 349, 2637, 148, 605, 2, 8003, 15, 123, 125, 68, 2, 6853, 15, 349, 165, 4362, 98, 5, 4, 228, 9, 43, 2, 1157, 15, 299, 120, 5, 120, 174, 11, 220, 175, 136, 50, 9, 4373, 228, 8255, 5, 2, 656, 245, 2350, 5, 4, 9837, 131, 152, 491, 18, 2, 32, 7464, 1212, 14, 9, 6, 371, 78, 22, 625, 64, 1382, 9, 8, 168, 145, 23, 4, 1690, 15, 16, 4, 1355, 5, 28, 6, 52, 154, 462, 33, 89, 78, 285, 16, 145, 95])
#  list([1, 14, 47, 8, 30, 31, 7, 4, 249, 108, 7, 4, 5974, 54, 61, 369, 13, 71, 149, 14, 22, 112, 4, 2401, 311, 12, 16, 3711, 33, 75, 43, 1829, 296, 4, 86

## 由于限制，所以最大不会超过10000

## interesting code 解码评论为英文单词

# word_index = imdb.get_word_index()
# ## tips use "items()" because python has abondoned "item"
# reverse_word_index =dict([(value,key) for (key,value) in word_index.items()])
# decoded_review = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])
# print(decoded_review)

##

import   numpy as np

def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i , sequence in enumerate(sequences):
        results[i,sequence]=1. #将数据相应指定索引为1.(.表示浮点数)
    #array索引维度之间用,
    # for i, sequence in enumerate(train_data):
    #     print(sequence)
    #这里sequence其实是数组,相当与result[i,sequnce[0]]=result[i,sequnce[1]]......=1
    return results
#将训练测试数据向量化
x_train = vectorize_sequences(train_data)
x_test =vectorize_sequences(test_data)
#标签向量化 测试分类本身就是01形
y_train= np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers

model= models.Sequential()
#参数16为隐藏单元
#inputshape是每一条数据的限制，x_test ——>[25000,10000],作用的是后面的10000（特征）
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#类似relu的激活函数能够引入非线性变换

# model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#配置优化器
from keras import optimizers
# model.compile(optimizer=optimizers.rmsprop_v2.RMSProp(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])

#使用自定义的损失和指标
from keras import losses
from keras import metrics

# model.compile(optimizer=optimizers.rmsprop_v2.RMSProp(lr=0.001),loss=losses.binary_crossentropy,metrics=[metrics.binary_accuracy])
model.compile(optimizer=optimizers.rmsprop_v2.RMSProp(lr=0.001),loss='mse',metrics=[metrics.binary_accuracy])

#留出验证集
#validdation_data 为验证参数
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]
#可以通过截取数据来作为验证集

# history= model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
history= model.fit(x_train,y_train,epochs=40,batch_size=512,validation_data=(x_test,y_test))
print(history.history.values())
results= model.evaluate(x_test,y_test)
print(results)
# history 中的histroy成员是一个字典：dict_keys(['loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy'])
# val_前缀为验证损失和精度

#做两个图

import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
#损失

plt.interactive(True)
history_dict= history.history
loss_values = history_dict['loss']
val_loss_values= history_dict['val_loss']
epochs= range(1,len(loss_values)+1)
plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.legend()
plt.savefig("temp.png",dpi=200)
plt.show()
exit()
#精度
# plt.clf()
# acc=history_dict['binary_accuracy']
# val_acc=history_dict['val_binary_accuracy']
# plt.plot(epochs,acc,'bo',label='Training acc')
# plt.plot(epochs,val_acc,'b',label='Validation acc')
# plt.title('Training and validdation accuracy')
# plt.xlabel('Epoches')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.show()

##

#其实验证损失比较大，可能存在问题
