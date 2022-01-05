## pytorch
``` bash
conda create -n pytorch python=3.9

conda install -n pytorch pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## add libcudart 11.0 
``` bash
sudo vim /etc/ld.so.conf

include /etc/ld.so.conf.d/*.conf
/home/wufisher/anaconda3/envs/pytorch/lib/

export LIBRARY_PATH=$LIBRARY_PATH:/home/wufisher/anaconda3/envs/pytorch/lib/ 
or if install all of the cuda   

export LD_LIBRARY_PATH=/usr/local/cuda-11.5/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

## jupyter
```bash
conda install -c conda-forge -n pytorch jupyterlab
conda activate pytorch 
conda install (-n pytorch) ipykernel
python -m ipykernel install --user --name pytorch

to start : bash(pytorch)>>jupyter lab
```

## vscode 
autocompete
  ```bash
  setting >>Typeshed Paths
  additem >>~/anaconda3/envs/pytorch/lib

  ```