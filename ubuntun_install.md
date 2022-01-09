# for ubuntu

## anaconda
    官网或者清华镜像源下载，
```linux
    sudo bash .... .sh
```
``` bash
vim ./bashrc

export PATH=/opt/anaconda/bin:$PATH
```
```bash
conda config --set show_channel_urls yes
```

```bash
vim ./condarc

channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/main
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/r
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.bfsu.edu.cn/anaconda/cloud
  msys2: https://mirrors.bfsu.edu.cn/anaconda/cloud
  bioconda: https://mirrors.bfsu.edu.cn/anaconda/cloud
  menpo: https://mirrors.bfsu.edu.cn/anaconda/cloud
  pytorch: https://mirrors.bfsu.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.bfsu.edu.cn/anaconda/cloud
```



## tensorflow
``` bash
sudo apt install nvidia-cuda-toolkit

conda create -n tensorflow python=3.9

conda install -n tensorflow -c 

conda-forge cudatoolkit

conda install -n tensorflow -c 

conda-forge cudnn

pip install tensorflow_gpu


```

## pytorch
``` bash
conda create -n pytorch python=3.9

conda install -n pytorch pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## mxnet
```bash
conda create -n d2l python=3.8
pip install mxnet-cu112
pip install d2l

```

## add libcudart 11.0 
``` bash
sudo vim /etc/ld.so.conf

include /etc/ld.so.conf.d/*.conf
/home/wufisher/anaconda3/envs/tensorflow/lib/

export LIBRARY_PATH=$LIBRARY_PATH:/home/wufisher/anaconda3/envs/tensorflow/lib/ 

```
## libnccl
```bash
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

sudo apt update
sudo apt install libnccl2 libnccl-dev
test:
python
import mxnet as mx
mx.context.num_gpus()
```



## chinese input wrong(pycharm1)
    add vmooption
    （help--VMoption）
```commandline
-Drecreate.x11.input.method=true
```
    restart pycharm
## jupyter
```
conda install -c conda-forge -n tensorflow jupyterlab
conda activate tensorflow 
conda install (-n tensorflow) ipykernel
python -m ipykernel install --user --name tensorflow

to start : bash(tensorflow)>>jupyter lab
```

## vscode 
    autocompete
  ```
  setting >>Typeshed Paths
  additem >>~/anaconda3/envs/tensorflow/lib
  additem >>~/anaconda3/envs/pytorch/lib

  ```
