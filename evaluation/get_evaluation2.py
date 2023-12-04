'''
    混淆矩阵
    Recall、Precision、MIOU计算
'''
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import cv2

# 输入必须为灰度图
# labels为你的像素值的类别
from utils import keep_image_size_open

# def get_dice(y_true, y_pred):
#     smooth = 1
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     intersection = np.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
from sklearn.metrics import f1_score

def get_dice(l,p):
    smooth=1e-5
    return (2. * np.sum(l * p)+smooth) / (smooth + p.sum() + l.sum())


def get_iou(l, p):
    # print(p.shape,l.shape)
     return (1e-5 + np.sum(l * p)) / (1e-5 + p.sum() + l.sum() - np.sum(l * p))


if __name__ == '__main__':
    from PIL import Image
    label_path="/home/ouyang/milu_Unet/data/test/labels/"
    pred_path="/home/ouyang/milu_Unet/result/"

    sum_MIOU=0
    sum_dice=0
    f1_scores=0

    list1=[]
    list2=[]


    t_list=[]

    for file in os.listdir(pred_path):
        # print(file)
        label = keep_image_size_open(label_path+file)
        pred = Image.open(pred_path+file)

        # print(type(label))
        # print(type(pred))
        #
        # print(label)
        # print(pred)
        l, p = np.array(label).astype(int), np.array(pred).astype(int)
        p=p/255
        p[p>0]=1
        l[l>0]=1
        # plt.subplot(1,2,1)
        # plt.imshow(l)
        # plt.subplot(1,2,2)
        # plt.imshow(p)
        # plt.show()
        MIOU=get_iou(l,p)
        dice=get_dice(l,p)

        list1.append(MIOU)
        list2.append(dice)

        print(MIOU,dice)
        sum_MIOU=sum_MIOU+MIOU
        sum_dice=sum_dice+dice

        temp_l=torch.flatten(torch.from_numpy(l))
        temp_p=torch.flatten(torch.from_numpy(p))

        # f1_scores=f1_scores+f1_score(temp_l,temp_p)

    Average_MIOU=sum_MIOU/len(os.listdir(label_path))
    print(Average_MIOU)
    Average_dice = sum_dice/ len(os.listdir(label_path))
    print(Average_dice)
    # Average_f1_score=f1_scores/len(os.listdir(label_path))
    # print(Average_f1_score)

    iou_std=np.std(list1,ddof=1)
    print(iou_std)
    dice_std=np.std(list2,ddof=1)
    print(dice_std)
    print(list2)

