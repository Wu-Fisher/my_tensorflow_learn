from keras.models import load_model
# import tensorflow as tf
# 注意这个地址
model = load_model("/home/wufisher/my_tensorflow_learn/python/tensorflow/day10/candd_v2.h5")
print(model.summary())
exit()