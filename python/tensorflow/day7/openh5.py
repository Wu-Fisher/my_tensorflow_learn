import h5py
f=h5py.File('candd_v1.h5','r')
print(f.get('optimizer_weights').get('RMSprop'))