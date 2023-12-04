import os
import cv2
import numpy as np
import SimpleITK as sitk
from augmentation import *
from skimage import measure
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import cv2
import random
from scipy import ndimage

#随机进行水平翻转或者垂直翻转,核心是cv2.flip()函数
def random_flip(img_input,lab_input):#对称

    img=img_input.copy()
    lab=lab_input.copy()
    i=np.random.randint(2)
    image = cv2.flip(img, i)
    label=cv2.flip(lab,i)
    return image,label

#水平翻转,核心是cv2.flip()函数
def random_flip_1(img_input,lab_input):#

    img=img_input.copy()
    lab=lab_input.copy()
    # i=np.random.randint(2)
    image = cv2.flip(img, 1)
    label=cv2.flip(lab,1)
    return image,label

#垂直翻转,核心是cv2.flip()函数
def random_flip_0(img_input,lab_input):

    img=img_input.copy()
    lab=lab_input.copy()
    # i=np.random.randint(2)
    image = cv2.flip(img, 0)
    label=cv2.flip(lab,0)
    return image,label

#仿射变换,核心是cv2.warpAffine()函数
def random_thanslate(img_input,lab_input):
    img = img_input.copy()
    lab = lab_input.copy()
    i = np.random.randint(2)
    if i==0:
        M=np.float32([[1,0,0],
                    [0,1,50]])
    else:
        M=np.float32([[1,0,0],
                   [0,1,-50]])
    image=cv2.warpAffine(img,M,(img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR)
    label=cv2.warpAffine(lab,M,(lab.shape[1],lab.shape[0]),flags=cv2.INTER_LINEAR)
    return image,label

#随机旋转,核心是cv2.warpAffine()函数
def random_rotation(img_input,lab_input):
    img = img_input.copy()
    lab=lab_input.copy()
    H, W ,c= img_input.shape
    center = (W/2, H/2)
    angle = np.random.randint(0, 360)
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    image = cv2.warpAffine(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,  borderValue=(0))
    label=cv2.warpAffine(lab, M, (lab.shape[1],lab.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,  borderValue=(0))
    return image,label

#resize
def random_crop(img_input,lab_input):
    img = img_input.copy()
    lab = lab_input.copy()
    H, W ,c= img_input.shape

    l=300
    min_x=np.random.randint(H-l)
    max_x=min_x+l
    min_y=np.random.randint(W-l)
    max_y=min_y+l
    img=img[min_x:max_x,min_y:max_y]
    lab = lab[min_x:max_x, min_y:max_y]

    img_resize=cv2.resize(img,(512,512),interpolation=cv2.INTER_LINEAR)
    lab_resize=cv2.resize(lab,(512,512),interpolation=cv2.INTER_LINEAR)
    return img_resize,lab_resize

#添加高斯噪声
def random_gauss(img_input, lab_input):
    aug = iaa.GaussianBlur(sigma=(0, 2.0))
    im_aug, seg_aug = aug(image=img_input, segmentation_maps=lab_input)
    return im_aug, seg_aug

# def random_aug(img_input, lab_input):
#     # img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
#     # lab_input = np.expand_dims(lab_input, axis =0).astype(np.int32)
#     # img_input = np.expand_dims(img_input, axis=3).astype(np.float32)
#     # lab_input = np.expand_dims(lab_input, axis =3).astype(np.int32)
#     # print(lab_input.shape)
#     # print(type(lab_input))
#     lab_input=lab_input.astype(np.bool_)
#     # print(img_input.shape)
#     # img_input = np.expand_dims(img_input, axis=0)
#     lab_input = np.expand_dims(lab_input, axis =0)
#     img_input = np.expand_dims(img_input, axis=3)
#     lab_input = np.expand_dims(lab_input, axis =3)
#     img_input = np.array(img_input*255).astype(np.uint8)
#     sometimes = lambda aug: iaa.Sometimes(p=0.5, then_list=aug)
#     ####定义一个lambda表达式，可以以p的概率去执行sometimes传递的图像增强
#
#     seq = iaa.Sequential(
#         [
#             sometimes(iaa.Crop(percent=(0, 0.1), keep_size=True)),  ####以p=0.5的概率去执行图片裁剪
#             sometimes(iaa.Affine(
#                 scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#                 translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#                 rotate=(-45, 45),
#                 shear=(-16, 16),
#                 order=[0, 1],
#                 cval=(0, 255),
#                 mode=ia.ALL
#             )),  ######order = 0 最临近插值 =1双线性插值
#             ####mode取值 * "constant" -> cv2.BORDER_CONSTANT
#             # * "edge" -> cv2.BORDER_REPLICATE
#             # * "symmetric" -> cv2.BORDER_REFLECT
#             # * "reflect" -> cv2.BORDER_REFLECT_101
#             # * "wrap" -> cv2.BORDER_WRAP
#             ########If ia.ALL, a value from the discrete range [0 .. 255] will be sampled per image。cval图像变化后padding的值
#             iaa.SomeOf((0, 5),  ####从下列序列中挑选0-5个图像增强的方法 相当与random.choice,下面可以放些自认为不太重要的方法
#                        [
#                            sometimes(
#                                iaa.Superpixels(
#                                    p_replace=(0, 1.0),
#                                    n_segments=(20, 200)
#                                )
#                            ),  #######执行 SLIC 超像素分割算法，对于每副图片p_replace=0-1的概率替换像素值。生成n_segments个超像素值
#                            ######iaa.OneOf  ==  iaa.SomeOf(1,1)  下面3个方法选一个
#                            iaa.OneOf([
#                                iaa.GaussianBlur((0, 3.0)),
#                                iaa.AverageBlur(k=(2, 7)),
#                                iaa.MedianBlur(k=(3, 11)),
#                            ]),
#                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
#                            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
#                            sometimes(iaa.OneOf([
#                                iaa.EdgeDetect(alpha=(0, 0.7)),
#                                iaa.DirectedEdgeDetect(
#                                    alpha=(0, 0.7), direction=(0.0, 1.0)
#                                ),
#                            ])),
#                            iaa.AdditiveGaussianNoise(
#                                loc=0, scale=(0.0, 0.05 *255), per_channel=0.5
#                            ),
#                            iaa.OneOf([
#                                iaa.Dropout((0.01, 0.1), per_channel=0.5),  ####丢掉1%-10%的像素信息
#                                iaa.CoarseDropout(
#                                    (0.03, 0.15), size_percent=(0.02, 0.05),
#                                    per_channel=0.2
#                                ),  #####丢掉矩形块内的像素信息
#                            ]),
#                            ####5%的概率进行255-像素值进行图像反化
#                            iaa.Invert(0.05, per_channel=True),
#                            # 每个像素加上一个-10到10的值
#                            iaa.Add((-10, 10), per_channel=0.5),
#                            ####灰度图  cv2图片通道顺序BGR
#                            # iaa.Grayscale(alpha=(0.0, 1.0), from_colorspace='BGR'),
#                            ######通过使用位移场在局部移动像素来转换图像。
#                            # 论文Simard, Steinkraus and Platt Best Practices for Convolutional Neural Networks applied to Visual Document Analysis in Proc. of the International Conference on Document Analysis and Recognition, 2003
#                            sometimes(
#                                iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
#                            ),
#                            #####图像上设置规则网格，并通过仿射变换随机移动这些点的邻域进行图片局部扭曲。
#                            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
#                        ],
#                        # do all of the above augmentations in random order
#                        random_order=True
#                        )
#         ],
#         # do all of the above augmentations in random order
#         random_order=True
#     )
#     images_aug, segmaps_aug = seq(images=img_input, segmentation_maps=lab_input)
#     images_aug = np.squeeze(images_aug, axis=0)
#     segmaps_aug = np.squeeze(segmaps_aug, axis=0)
#     images_aug = np.squeeze(images_aug, axis=2)
#     segmaps_aug = np.squeeze(segmaps_aug, axis=2)
#     segmaps_aug=segmaps_aug.astype(np.uint8)
#     images_aug = np.array(images_aug/255).astype(np.float32)
#     # print("image:",images_aug.shape,images_aug.max(),images_aug.min())
#     # print("segmap:",segmaps_aug.shape, segmaps_aug.max(),segmaps_aug.min())
#     return images_aug, segmaps_aug

def random_rotate(x):
    angle = random.randint(-20,20)
    axes = random.sample((0,1,2),k = 2)
    rotated_x = ndimage.rotate(x, angle = angle, axes = axes, reshape = False, order = 3, mode = 'constant')
    return rotated_x

def random_shift(x):
    'random shift image along three axes within range [-10, 10]'
    # define random shift value
    shift = [random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)]

    # perform shift
    shifted_x = ndimage.shift(x, shift=shift, order=3, mode='constant', cval=0.0)
    return shifted_x

def random_zoom_and_crop(x):
    'random zoom image along three axes within range [0.9, 1.1]'
    # define zoom value
    zoom = [random.uniform(0.9, 1.1), random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)]

        # perform zoom
    zoomed_x = ndimage.zoom(x, zoom, order=3, mode='constant', cval=0.0)

        # padding and crop
    orig_shape = x.shape
    zoom_shape = zoomed_x.shape

    if zoom[0] < 1:
        padding_width = orig_shape[0] - zoom_shape[0]
        z0 = np.zeros((padding_width, zoomed_x.shape[1], zoomed_x.shape[2]), dtype=x.dtype)
        x_x = np.concatenate((zoomed_x, z0), axis=0)
    else:
        seed = random.randint(0, zoom_shape[0] - orig_shape[0])
        x_x = zoomed_x[seed:seed + orig_shape[0]][:][:]

    if zoom[1] < 1:
        padding_width = orig_shape[1] - zoom_shape[1]
        z0 = np.zeros((x_x.shape[1], padding_width, x_x.shape[2]), dtype=x.dtype)
        x_y = np.concatenate((x_x, z0), axis=1)
    else:
        seed = random.randint(0, zoom_shape[1] - orig_shape[1])
        x_y = x_x[:][seed:seed + orig_shape[1]][:]

    if zoom[2] < 1:
        padding_width = orig_shape[2] - zoom_shape[2]
        z0 = np.zeros((x_y.shape[0], x_y.shape[1], padding_width), dtype=x.dtype)
        x_z = np.concatenate((x_y, z0), axis=2)
    else:
        seed = random.randint(0, zoom_shape[0] - orig_shape[0])
        x_z = x_y[:][:][seed:seed + orig_shape[0]]

    return x_z


def train_img_lab(root_path):
    data_name_list = os.listdir(os.path.join(root_path, 'train', 'imgs'))
    data_name_list.sort()
    print(data_name_list)
    idx=0
    for data_name in data_name_list:
        img_data = []
        lab_data = []
        iter_img=cv2.imread(os.path.join(root_path, 'train', 'imgs',data_name))/255
        iter_lab=cv2.imread(os.path.join(root_path, 'train', 'gts',data_name),0)/255

        img_data.append(iter_img)
        lab_data.append(iter_lab)

        for i in range(0, 8):
            img_random_ration, lab_random_ration = random_rotation(iter_img, iter_lab)
            img_data.append(img_random_ration)
            lab_data.append(lab_random_ration)
            img_random_crop, lab_random_crop = random_crop(iter_img, iter_lab)
            img_data.append(img_random_crop)
            lab_data.append(lab_random_crop)

        if iter_lab.max() > 0:
            img_flip_0, lab_flip_0 = random_flip_0(iter_img, iter_lab)
            img_data.append(img_flip_0)
            lab_data.append(lab_flip_0)

            img_flip_1, lab_flip_1 = random_flip_1(iter_img, iter_lab)
            img_data.append(img_flip_1)
            lab_data.append(lab_flip_1)

            img_thanlate, lab_thanlate = random_thanslate(iter_img, iter_lab)
            img_data.append(img_thanlate)
            lab_data.append(lab_thanlate)

            # for i in range(0, 5):
            #     # print(iter_lab.max(), iter_lab.dtype, iter_lab.shape)
            #     img_random, lab_random = random_aug(iter_img, iter_lab)
            #     img_data.append(img_random)
            #     lab_data.append(lab_random)
        for img ,mask in zip(img_data,lab_data):
            cv2.imwrite(os.path.join(root_path, 'train_ag', 'imgs',str(idx)+".png"),img*255)
            cv2.imwrite(os.path.join(root_path, 'train_ag', 'gts',str(idx)+".png"),mask*255)
            idx+=1

if __name__=="__main__":
    train_img_lab("/home/ouyang/data_augmentation/HeiSurF/")                                          
