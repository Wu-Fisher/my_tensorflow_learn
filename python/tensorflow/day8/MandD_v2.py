

train_dir='/home/wufisher/dataset_m/mm_and_dd/cats_and_dogs_small/train'
validation_dir='/home/wufisher/dataset_m/mm_and_dd/cats_and_dogs_small/validation'
test_dir='/home/wufisher/dataset_m/mm_and_dd/cats_and_dogs_small/test'
train_cats_dir='/home/wufisher/dataset_m/mm_and_dd/cats_and_dogs_small/train/cats'
from keras.preprocessing.image import ImageDataGenerator

# datagen= ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

from keras.preprocessing import image
import os
# fames= [os.path.join(train_cats_dir,fname) for fname in os.listdir(train_cats_dir)]
# img_path = fnames[3]
# img= image.load_img(img_path,target_size=(150,150))
# x= image.img_to_array(img)
# x=x.reshape((1,)+x.shape)
# i=0
# for batch in datagen.flow(x,batch_size=1):
#     plt.figure(i)
#     imgplot=plt.imshow(image.array_to_img(batch[0]))
#     i+=1
#     if i%4==0:
#         break
#
# plt.savefig('test.png')
# plt.show()
#
from keras import  models
from keras import layers
from keras import optimizers
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.rmsprop_v2.RMSProp(lr=1e-4),
              metrics=['acc'])
train_datagen= ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
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
model.save('candd_v2.h5')

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
plt.savefig('acc_v2.png')

plt.figure()

plt.plot(range(1,len(loss)+1),loss,'bo',label='Training loss')
plt.plot(range(1,len(val_loss)+1),val_loss,'b',label='VA loss')
plt.title('LOSS')
plt.legend()
plt.savefig('loss_v2.png')
plt.show()



exit()