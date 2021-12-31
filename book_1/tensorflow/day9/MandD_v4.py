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
conv_base.trainable=True
set_tr = False
for layer in conv_base.layers:
    if layer.name=='block5_conv1':
        set_trainable=True
    if set_tr:
        layer.trainable=True
    else:
        layer.trainable=False
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
model.save('candd_v4.h5')
print("history:")
print(history.history)
import matplotlib.pyplot as plt

acc=history.history['acc']
val_acc =history.history['val_acc']
loss= history.history['loss']
val_loss= history.history['val_loss']
epochs= range(1,len(acc)+1)

## 让曲线更平滑
def smooth_curve(points,factor=0.8):
    smoothed_points=[]
    for point in points:
        if len(smoothed_points)!=0:
            previous=smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points



plt.plot(epochs,smooth_curve(acc),'bo',label='Training acc')
plt.plot(range(1,len(val_acc)+1),smooth_curve(val_acc),'b',label='VA acc')
plt.title('ACC')
plt.legend()
plt.savefig('acc_v4.png')

plt.figure()

plt.plot(range(1,len(loss)+1),smooth_curve(loss),'bo',label='Training loss')
plt.plot(range(1,len(val_loss)+1),smooth_curve(val_loss),'b',label='VA loss')
plt.title('LOSS')
plt.legend()
plt.savefig('loss_v4.png')
plt.show()



exit()