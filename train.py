import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from archs  import UNext


from torchvision.utils import save_image

device = "cuda"
weight_path = 'params/mynet.pth'
data_path = r'data'
save_path = 'train_image'
val_data_path="/home/ouyang/milu_Unet/data/val/"

#
# class data_prefetcher():
#     def __init__(self, loader):
#         self.loader = iter(loader)
#         # self.stream = torch.cuda.Stream()
#         self.preload()
#
#     def preload(self):
#         try:
#             self.next_data = next(self.loader)
#         except StopIteration:
#             self.next_input = None
#             return
#         # with torch.cuda.stream(self.stream):
#         #     self.next_data = self.next_data.cuda(non_blocking=True)
#
#     def next(self):
#         # torch.cuda.current_stream().wait_stream(self.stream)
#         data = self.next_data
#         self.preload()
#         return data

if __name__ == '__main__':
    num_classes = 1 + 1  # +1是背景也为一类
    data_loader = DataLoader(MyDataset(data_path), batch_size=64, shuffle=True,num_workers=64)
    val_dataloader = DataLoader(MyDataset(val_data_path), batch_size=12, shuffle=True,num_workers=12)####eval  dataloader

    net = UNext(num_classes).to("cuda")
    net = nn.DataParallel(net,device_ids=[0,1,2,3])
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')
    opt = optim.Adam(net.parameters())
    loss_fun = nn.CrossEntropyLoss()
    epoch = 1
    while epoch <=1000:
        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
            image, segment_image = image.to(device).float(), segment_image.to(device).float()
            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image.long())
            # print(train_loss.item())
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            # if i % 1 == 0:
                # print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            _image = image[0]
            _segment_image = torch.unsqueeze(segment_image[0], 0) * 255
            _out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255

            img = torch.stack([_segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png')
        if epoch % 10 == 0:
            torch.save(net.state_dict(), weight_path)
            print('save successfully!')
        net.eval()
        val_loss=0
        for i, (image, segment_image) in enumerate(tqdm.tqdm(val_dataloader)):
            image, segment_image = image.to(device).float(), segment_image.to(device).float()
            out_image = net(image)
            val_loss = loss_fun(out_image, segment_image.long())
            val_loss+=val_loss.item()
        print("----val loss:",val_loss/len(val_dataloader))
        net.train()
        epoch += 1
