# for ubuntu

## anaconda
    官网或者清华镜像源下载，
```linux
    sudo bash .... .sh
```
export PATH=/opt/anaconda/bin:$PATH

```
conda config --set show_channel_urls yes

```

```
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
```
sudo apt install nvidia-cuda-toolkit
conda create -n tensorflow python=3.9
conda install -n tensorflow -c conda-forge cudatoolkit
conda install -n tensorflow -c conda-forge cudnn
pip install tensorflow_gpu


```

## add libcudart 11.0 
```
sudo vim /etc/ld.so.conf

include /etc/ld.so.conf.d/*.conf
/home/wufisher/anaconda3/envs/tensorflow/lib/

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

to start : bash>>jupyter lab
```