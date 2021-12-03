import os
from PIL import Image
folder_path = '/home/wufisher/dataset_m/mm_and_dd/cats_and_dogs_small/train/dogs'  #写入你图片所在的文件夹，即包含该图片的文件夹
extensions = []
for filee in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filee)
    print('** Path: {}  **'.format(file_path), end="\r", flush=True)
    im = Image.open(file_path)
    rgb_im = im.convert('RGB')
    if filee.split('.')[1] not in extensions:
        extensions.append(filee.split('.')[1])
