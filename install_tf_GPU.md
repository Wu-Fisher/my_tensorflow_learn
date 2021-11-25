1.install anaconda (if win add path environment)
2.change channel_url
    for conda
    1.cmd:  conda config --set show_channel_urls yes
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
3.environment
    cmd: conda create -n tensorflow python=3.9
    then remember do everything after 
        cmd: activate tensorflow
4.install cuda and cudnn
    cuda:
        conda install -c conda-forge cudatoolkit
    cudnn
        conda install -c conda-forge cudnn
    (from conda-forge get the newest version)

5. install tensorflow_gpu
    pip install tensorflow-gpu==2.6.0 
    (2021/11/25)

6.check 
    cmd: python
    cmd: import tensorflow as tf
    make a eszy test

    cmd: tf.test.is_gpu_available()
    cmd: tf.config.list_physical_devices('GPU')

## for vscode win
    change the console from powershell to cmd
    and remember activate tensorflow