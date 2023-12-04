import thop
import torch
from torchsummary import summary
import numpy as np
from thop import profile
from thop import clever_format
from  deeplab import DeepLabv3_plus
from archs_original_unext import Original_UNext
from net import UNet
from  archs import UNext
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

input=torch.randn(1,3,512,512)
# model=UNext(2)
#
model=Original_UNext(2)

flops,params=thop.profile(model,inputs=(input,))
print(flops/1000000)
print(params/1000000)
#?,params=clever_format([flops,params],"%.3f")

#
# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model=UNext(2).to(device)
#
# summary(model,(3,512,512))



