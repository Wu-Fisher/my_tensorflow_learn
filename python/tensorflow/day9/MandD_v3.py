train_dir='/home/wufisher/dataset_m/mm_and_dd/cats_and_dogs_small/train'
validation_dir='/home/wufisher/dataset_m/mm_and_dd/cats_and_dogs_small/validation'
test_dir='/home/wufisher/dataset_m/mm_and_dd/cats_and_dogs_small/test'
train_cats_dir='/home/wufisher/dataset_m/mm_and_dd/cats_and_dogs_small/train/cats'
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import  vgg16
from keras import  models
from keras import layers
from keras import optimizers
conv_base= vgg16.VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
## very important below
conv_base.trainable=False
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.rmsprop_v2.RMSProp(lr=2e-5),
              metrics=['acc'])
train_datagen= ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
## 不能增强训练数据
test_datagen=ImageDataGenerator(rescale=1./255)

train_generator =train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

validation_generator=test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
history =model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)
model.save('candd_v3.h5')

import matplotlib.pyplot as plt

acc=history.history['acc']
val_acc =history.history['val_acc']
loss= history.history['loss']
val_loss= history.history['val_loss']

epochs= range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(range(1,len(val_acc)+1),val_acc,'b',label='VA acc')
plt.title('ACC')
plt.legend()
plt.savefig('acc_v3.png')

plt.figure()

plt.plot(range(1,len(loss)+1),loss,'bo',label='Training loss')
plt.plot(range(1,len(val_loss)+1),val_loss,'b',label='VA loss')
plt.title('LOSS')
plt.legend()
plt.savefig('loss_v3.png')
plt.show()



exit()