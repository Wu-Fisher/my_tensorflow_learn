from keras.datasets import imdb
(train_data,train_labels),(test_data,test_labels)= imdb.load_data(num_words=10000)
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
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=4,batch_size=512)
results= model.evaluate(x_test,y_test)
print(results)
## 预测结果
print(model.predict(x_test))

#记得退出
exit()