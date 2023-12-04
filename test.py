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
from PIL import Image
import ResUnet_model

net=UNet(2).cuda()
# net = DeepLabv3_plus().to("cuda")
# net = UNext(2).to("cuda")
# net = Original_UNext(2).to("cuda")
# net= ResUnet_model.ResU net(3).to("cuda")
# net = NestedUNet(3,2).to("cuda")
# net = AttU_Net().to("cuda")
# net = mobilenet_unet.Mobilenet_Unet(3, 2).to("cuda")

net=nn.DataParallel(net,device_ids=[7])
weights='params/unet.pth'
# weights='params/mynet.pth'
# weights="params/attention_unet.pth"
# weights="params/unetplusplus.pth"
# weights="params/resunet.pth"
# weights="params/originalunext.pth"
# weights="params/ourunext.pth"
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully')
else:
    print('no loading')

# _input=input('please input JPEGImages path:')

path="/home/ouyang/milu_Unet/data/test/images/"
# path="/home/ouyang/milu_Unet/data/JPEGImages/"
# path="/home/ouyang/ffpmeg/frame/"
net.eval()
save_path="/home/ouyang/milu_Unet/result/"
# save_path="/home/ouyang/ffpmeg/unet_result/"
import shutil

if(os.path.exists(save_path)):
    shutil.rmtree(save_path)
    os.mkdir(save_path)

count=0
time1=time.time()
for file in os.listdir(path):

    img=keep_image_size_open_rgb(path+file)
    img=np.array(img) / 255
    img_data=transform(img).cuda()
    img_data=torch.unsqueeze(img_data,dim=0).float()
    # img_data = torch.unsqueeze(img_data/255, dim=0)
    start_time = time.time()
    out=net(img_data)
    end_time=time.time()

    out=torch.argmax(out,dim=1)
    out=torch.squeeze(out,dim=0)
    out=out.unsqueeze(dim=0)
    print(set((out).reshape(-1).tolist()))

    out=(out).permute((1,2,0)).cpu().detach().numpy()
    # cv2.imwrite(save_path + file, out )
    cv2.imwrite(save_path+file,out*255)
    count=count+1
    # cv2.imshow('out',out*255.0)
    # cv2.waitKey(0)
time2=time.time()
print(time2-time1)
print(count)
