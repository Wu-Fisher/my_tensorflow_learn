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




