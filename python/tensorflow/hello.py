import tensorflow as tf
import os
import numpy as np
#to abandon the message of no set
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# tf.test.is_gpu_available()
# tf.config.list_physical_devices('GPU')

a=np.zeros((10))
b=[2,3,6]
a[b]=1
print(a)