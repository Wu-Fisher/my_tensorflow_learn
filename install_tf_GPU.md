## 1.install anaconda 

(if win add path environment)

for manjaro

```linux
vim ~/.bashrc
export PATH=$PATH:【你的安装目录】
source ~/.bashrc
source /opt/anaconda/bin/activate root 
```

#have evns	NOOOOO in linux

```linux
sudo cp -r /run/media/wufisher/5F47-73FA/envs/tensorflow  /opt/anaconda/envs/

```

then check

```linux
conda info --envs 
```



## 2.change channel_url

###     for win

​    1.cmd:  conda config --set show_channel_urls yes

    2.  open the .condarc (/administrator1)
        3. replace 
        channels:
    - defaults
        show_channel_urls: true
        channel_alias: https://mirror.tuna.tsinghua.edu.cn/anaconda
        default_channels:
        
        - https://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/main
        
        - https://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/free
        
        - https://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/r
        
        - https://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
        
        - https://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
          custom_channels:
          conda-forge: https://mirror.tuna.tsinghua.edu.cn/anaconda/cloud
      msys2: https://mirror.tuna.tsinghua.edu.cn/anaconda/cloud
      bioconda: https://mirror.tuna.tsinghua.edu.cn/anaconda/cloud
      menpo: https://mirror.tuna.tsinghua.edu.cn/anaconda/cloud
      pytorch: https://mirror.tuna.tsinghua.edu.cn/anaconda/cloud
      simpleitk: https://mirror.tuna.tsinghua.edu.cn/anaconda/cloud
        4.cmd: conda clean -i
        for python
        cmd: pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
      
      ### forLinux
      
      (not command plz remake)
      
      ```text
      conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
      conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
      conda config --set show_channel_urls yes
      source /opt/anaconda/bin/activate root
      ```
      
      #### change the .condarc
      
      1.vim ~/.condarc
      
      2.replace
      
      ```vim
      channels:
        - https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
        - https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
        - defaults
      show_channel_urls: true
      ```
      
      pytorch like conda-forge change like above


​      

## 3.environment

​    cmd: conda create -n tensorflow python=3.9
​    then remember do everything after 
​        cmd:conda activate tensorflow

        OR : source activate tensorflow
## MUST CHECK THE VERSION

conda install -n env_name 

## conda install -c conda-forge tensorflow-gpu

version maybe old than pip

## 4.install cuda and cudnn

remember sudo 

###     cuda:

​        conda install -c conda-forge cudatoolkit

###     cudnn:

​    conda install -c conda-forge cudnn
​    (from conda-forge get the newest version)

## 5.install tensorflow_gpu

pip install tensorflow-gpu==2.6.0 
(2021/11/25)

(linux just no (==2.6.0) will get the newest)

## 6.check 

​    cmd: python
​    cmd: import tensorflow as tf
​    make a eszy test

    cmd: tf.test.is_gpu_available()
    cmd: tf.config.list_physical_devices('GPU')

## for vscode win
    change the console from powershell to cmd
    and remember activate tensorflow