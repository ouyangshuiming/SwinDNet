import os
import time

import cv2
import numpy as np
import torch

import mobilenet_unet
from archs import UNext
from archs_original_unext import Original_UNext
from deeplab import DeepLabv3_plus
from demo_module import NestedUNet, AttU_Net
from net import *
from utils import *
from data import *
from torchvision.utils import save_image
from PIL import Image`
import ResUnet_model

# net=UNet(2).cuda()
# net = DeepLabv3_plus().to("cuda")
# net = UNext(2).to("cuda")
net = Original_UNext(2).to("cuda")
# net= ResUnet_model.ResUnet(3).to("cuda")
# net = NestedUNet(3,2).to("cuda")
# net = AttU_Net().to("cuda")
# net = mobilenet_unet.Mobilenet_Unet(3, 2).to("cuda")

net=nn.DataParallel(net,device_ids=[7])
# weights='params/unet.pth'
# weights='params/attention_unet.pth'
# weights='params/unetplusplus.pth'
# weights='params/resunet.pth'
weights='params/originalunext.pth'
# weights="params/ourunext.pth"
# weights="params/deeplabv3+.pth"
net.load_state_dict(torch.load(weights))

img=keep_image_size_open_rgb("/home/ouyang/milu_Unet/data/test/images/video01_00080_frame_83.png")
img=np.array(img) / 255
img_data=transform(img).cuda()
img_data=torch.unsqueeze(img_data,dim=0).float()
# img_data = torch.unsqueeze(img_data/255, dim=0)
start_time = time.time()
for i in range(1000):
      out=net(img_data)
end_time=time.time()
inference_time=(end_time-start_time)/1000
print(inference_time)