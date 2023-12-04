import os

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

from albumentations import (
    Resize, RandomCrop, HorizontalFlip, Normalize, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
import   random
class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]  # xx.png
        #SegmentationClass是标签图片
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        #JPEGImages是原图,由标签图片的路径得到原图图片的路径
        image_path = os.path.join(self.path, 'JPEGImages', segment_name)

        #label  resize
        segment_image = np.array(keep_image_size_open(segment_path))
        segment_image[segment_image>0]=1

        #image resize
        image = keep_image_size_open_rgb(image_path)
        image = np.array(image)

       #第二个数据集新加的f代码

        if(random.random()>0.1):
            image=random.uniform(0.8,1.2)*image
            image=image.astype(np.uint8)

        return transform(image/255), torch.Tensor(segment_image)

# if __name__ == '__main__':
#     from torch.nn.functional import one_hot
#     data = MyDataset('data')
#     print(data[0][0].shape)
#     print(data[0][1].shape)
#     out=one_hot(data[0][1].long())
#     print(out.shape)
