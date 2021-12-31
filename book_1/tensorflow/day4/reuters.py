from keras.datasets import reuters
(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=10000)
# 每一个样本为单词索引列表
# 标签为0～45的整数分类
##
#解码索引
# word_index=reuters.get_word_index()
# # print(word_index )
# reverse_word_index =dict([(value,key) for (key ,value)in word_index.items()])
# print(reverse_word_index)
# # decoded_newswire =' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])
# # #减去三是因为 0 1 2 为填充，序列开始，未知词保留索引——》是保留在train_data里面，不是在reverse_word里面
# # print(decoded_newswire)

import numpy as np

def vectorize_sequences(sequences,dimension=10000):
    results= np.zeros((len(sequences),dimension))
    for i ,sequence in enumerate(sequences):
        results[i,sequence]=1.
    return results

x_train= vectorize_sequences(train_data)
x_test= vectorize_sequences(test_data)

#标签也可以采用上述one-hot编码
#也可用自带的函数

from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

from keras import models
from keras import layers

model= models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
#有64种
#softmax指输出46种的概率分布
model.add(layers.Dense(46,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train =x_train[1000:]
y_val=one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history=model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
# print(history.history.keys())
#huitu
#损失 怒额
import  matplotlib.pyplot as plt

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()


##发现在epoch=9时过拟合，改进


exit()