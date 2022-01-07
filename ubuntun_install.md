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
  版本回退
```bash
  conda list --revision
  conda install --rev n
```
  删除环境
```bash
  conda remove -n nev --all

```



## tensorflow
``` bash
sudo apt install nvidia-cuda-toolkit

conda create -n tensorflow python=3.9

conda install -n tensorflow -c conda-forge cudatoolkit

conda install -n tensorflow -c conda-forge cudnn

pip install tensorflow_gpu


```
  test
```python
import tensorflow as tf
tf.test.is_gpu_available()
  tf.config.list_physical_devices('GPU')
```

## pytorch
``` bash
conda create -n pytorch python=3.9

conda install -n pytorch pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## add libcudart 11.0 
``` bash
sudo vim /etc/ld.so.conf

include /etc/ld.so.conf.d/*.conf
/home/wufisher/anaconda3/envs/tensorflow/lib/

export LIBRARY_PATH=$LIBRARY_PATH:/home/wufisher/anaconda3/envs/tensorflow/lib/ 
这个是cuda库的路径
/usr/local/cuda-11.5/lib64
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
## python
  1.like these below
```bash
  Error processing line 1 of /home/wufisher/anaconda3/envs/tensorflow/lib/python3.9/site-packages/google_auth-2.3.3-py3.10-nspkg.pth:

  Traceback (most recent call last):
    File "/home/wufisher/anaconda3/envs/tensorflow/lib/python3.9/site.py", line 169, in addpackage
      exec(line)
    File "<string>", line 1, in <module>
    File "<frozen importlib._bootstrap>", line 562, in module_from_spec
  AttributeError: 'NoneType' object has no attribute 'loader'

```
    去那个文件File，应该是第一行import和后面的语句没有换行，没有注意python的语法规范