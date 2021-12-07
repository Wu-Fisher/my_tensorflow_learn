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

```vim ./condarc
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