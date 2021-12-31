train_dir='/home/wufisher/dataset_m/mm_and_dd/cats_and_dogs_small/train'
validation_dir='/home/wufisher/dataset_m/mm_and_dd/cats_and_dogs_small/validation'
test_dir='/home/wufisher/dataset_m/mm_and_dd/cats_and_dogs_small/test'
from keras import layers
from keras import models
from PIL import Image
model= models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
# print(model.summary())

from keras  import  optimizers
model.compile(loss='binary_crossentropy',optimizer=optimizers.rmsprop_v2.RMSProp(lr=1e-4),metrics=['acc'])
## 图像数据预处理
#读取图像文件
#JPEG转为RGB像素
#转为浮点数张量
#缩放到0，1区间
##！！！都可以帮你完成

from keras.preprocessing.image import ImageDataGenerator
train_datagen =ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

##利用批量迭代生成器来处理

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

model.save('candd_v1.h5')
# print(history.history.keys())
# dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])
import matplotlib.pyplot as plt

acc=history.history['acc']
val_acc =history.history['val_acc']
loss= history.history['loss']
val_loss= history.history['val_loss']

epochs= range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='VA acc')
plt.title('ACC')
plt.legend()
plt.savefig('acc_v1.png')

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='VA loss')
plt.title('LOSS')
plt.legend()
plt.savefig('loss_v1.png')
plt.show()

exit()