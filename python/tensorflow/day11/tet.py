from keras.applications.vgg16 import VGG16
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
model = VGG16(weights='imagenet')

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input,decode_predictions
import numpy as np

img_path='/home/wufisher/dataset_m/h5inside/elephant.jpg'

img = image.load_img(img_path,target_size=(224,224))

x=image.img_to_array(img)

x=np.expand_dims(x,axis=0)

x=preprocess_input(x)

preds=model.predict(x)
print('Rredicted:',decode_predictions(preds,top=3)[0])

print(np.argmax(preds[0]))

# from tensorflow.keras import backend as K
from tensorflow.compat.v1.keras import backend as K
import tensorflow as tf
##注意修改递归深度
tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_v2_behavior()
import sys  # 导入sys模块
sys.setrecursionlimit(30000)  # 将默认的递归深度修改为3000
# print(model.output[:,386])
african_elephant_output = model.output[:,386]   # 预测向量中的非洲象元素

last_conv_layer = model.get_layer('block5_conv3')  # block5_conv3层的输出特征图，它是VGG16的最后一个卷积层

grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]   # 非洲象类别相对于block5_conv3输出特征图的梯度
# with tf.GradientTape() as gtape:
#      grads = gtape.gradient(african_elephant_output, last_conv_layer.output)
pooled_grads = K.mean(grads, axis=(0, 1, 2))   # 形状是（512， ）的向量，每个元素是特定特征图通道的梯度平均大小

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])  # 这个函数允许我们获取刚刚定义量的值：对于给定样本图像，pooled_grads和block5_conv3层的输出特征图

pooled_grads_value, conv_layer_output_value = iterate([x])  # 给我们两个大象样本图像，这两个量都是Numpy数组

for i in range(512):
     conv_layer_output_value[:, :, i] *= pooled_grads_value[i]  # 将特征图数组的每个通道乘以这个通道对大象类别重要程度

heatmap = np.mean(conv_layer_output_value, axis=-1)  # 得到的特征图的逐通道的平均值即为类激活的热力图

import matplotlib.pyplot as plt
plt.clf()
heatmap = np.maximum(heatmap,0)
heatmap /=np.max(heatmap)
#print(heatmap)
plt.matshow(heatmap)

import cv2
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
# 将热力图应用于原始图像
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# 这里的 0.4 是热力图强度因子
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('elephant_cam_last.jpg', superimposed_img)