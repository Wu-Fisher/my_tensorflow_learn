# For manjaro

## 1.yay -S anaconda 

```
vim ~/.bashrc
export PATH=$PATH:【你的安装目录】
like 
export PATH=/opt/anaconda/bin:$PATH

source ~/.bashrc
source /opt/anaconda/bin/activate root 
```

do the same  in .zshrc

```
conda init zsh (bash)
```

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
update  12.7
```
show_channel_urls: true
ssl_verify: true
channels:
  - defaults
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
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

if you dont have a100 or 3090

```
conda install -n tensorflow -c conda-forge cudatoolki=10.2
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

## 6.question

no 11.0 cudart

```linux
sudo find / -name 'libcudart.so.11.0'  
>>>/opt/anaconda/envs/tensorflow/lib/libcudart.so.11.0
sudo ldconfig /opt/anaconda/envs/tensorflow/lib/ 
//手动添加库文件 别忘了在运行 ldconfig -v(好像不用)
export LIBRARY_PATH=$LIBRARY_PATH:/opt/souanaconda/envs/tensorflow/lib/ 
AT THE SAME TIME
ADD IN ./bashrc  ./zshrc

```

