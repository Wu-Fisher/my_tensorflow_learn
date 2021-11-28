# For manjaro

## 1.yay -S anaconda 

```
vim ~/.bashrc
export PATH=$PATH:【你的安装目录】
source ~/.bashrc
source /opt/anaconda/bin/activate root 
```

do the same  in .zshrc

## 2.change channel_url


conda config --set show_channel_urls yes

update
tsinghua use http!!!!!!! no https
```
ssl_verify: true

channels:

  - defaults

show_channel_urls: true

channel_alias: http://mirrors.tuna.tsinghua.edu.cn/anaconda

default_channels:

  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main

  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free

  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r

  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro

  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2

custom_channels:

  conda-forge: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud

  msys2: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud

  bioconda: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud

  menpo: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud

  pytorch: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud

  simpleitk: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```



## 3.create environment

```
conda create -n tensorflow python=3.9
```

check use

```
conda info --envs
```

start and over

```
conda activate tensorflow
conda deactivate 
```

## 4.install package

```
conda install -n tensorflow -c conda-forge cudatoolkit
conda install -n tensorflow -c conda-forge cudnn
pip install tensorflow_gpu
```

sure in env tensorflow

## 5.check

```
>>python
>>import tensorflow as tf
```

or

```
tf.test.is_gpu_available()
tf.config.list_physical_devices('GPU')
```

