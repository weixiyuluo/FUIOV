
import torch
from torch import nn


class CovModel(nn.Module):
    def __init__(self):
        super(CovModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=10,kernel_size=9,stride=1),
            # nn.BatchNorm2d(10),    
            nn.ReLU(),
            nn.Conv2d(in_channels=10,out_channels=20,kernel_size=9,stride=1),
            nn.Flatten(),
            nn.Linear(2880,512),
            # nn.LayerNorm(512),
            nn.Linear(512,10)

        )

    def forward(self,input):
        return self.model(input)
        # x = self.model(input)
        #print(x.shape)                         #[64,10]  第一维是batchsize，最后一维是类别
        # return nn.functional.softmax(x,dim=-1)   #client所使用的损失函数中已经包含softmax()，不能再设置一遍



class GTRSBmodel(nn.Module):
    def __init__(self):
        super(GTRSBmodel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=20,kernel_size=9,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=20,out_channels=43,kernel_size=9,stride=1),
            nn.Flatten(),
            nn.Linear(43*14*14,43)


        )

    def forward(self,input):
        return self.model(input)

class Cifarmodel(nn.Module):
    def __init__(self, class_num=10):
        super(Cifarmodel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=7, stride=1, padding=2),
            # 参数1：分组数（通常设为通道数的因数，比如2或5）；参数2：通道数
            nn.GroupNorm(num_groups=2, num_channels=20), 
            nn.ReLU(),
            nn.AvgPool2d(2),
            
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=4, stride=1, padding=2, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=40), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(in_channels=40, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=64), 
            nn.Flatten(),
            
            nn.Linear(8*8*64, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, input):
        return self.model(input)



if __name__ == '__main__':
    item = GTRSBmodel(10)
    print(sum(p.numel() for p in item.parameters() if p.requires_grad))  #打印参数个数
    # item(torch.rand((1,3,32,32)))  #(batch_size, channels, height, width)
