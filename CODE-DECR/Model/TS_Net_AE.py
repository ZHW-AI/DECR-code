from turtle import forward
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34,resnet18
import timm
import torch


from .Shunted_Transformer.Reverse_ResNet import resnet18 as r18
from .MDSFA import MDSFA_block, MEE_module



class AE_T_r18_S_r18_(nn.Module):
    def __init__(self,num_heads = 8, sr_ratio=[16, 8, 4, 2],
                 upsample =[False,True,True]) -> None:
        super(AE_T_r18_S_r18_,self).__init__()
        # Teacher
        Teacher = resnet18(pretrained=True)
        self.T_block1 = nn.Sequential(
                        Teacher.conv1,
                        Teacher.bn1,
                        Teacher.relu,
                        Teacher.maxpool)
        
        self.T_layer1 = Teacher.layer1
        self.T_layer2 = Teacher.layer2
        self.T_layer3 = Teacher.layer3
        self.T_layer4 = Teacher.layer4

        # Students
        Student = r18(pretrained=False)
        self.S_layer1 = Student.layer4
        self.S_layer2 = Student.layer3
        self.S_layer3 = Student.layer2 
        #修改-解码-([128, 64, 56, 56])变成([128, 3, 224, 224])
        self.S_layer4 =  nn.Sequential(
                                    nn.Upsample(scale_factor=4, mode='bilinear'),
                                    nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(3),
                                    nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(3),
                                    # nn.ReLU(),
                                    nn.Sigmoid(),
                                    )  
    
        # self.MFA1 = MFA_block(64, 64, 0)
        # self.MFA2 = MFA_block(128, 64, 1)
        # self.MFA3 = MFA_block(256, 128, 1)
        # self.DEE = DEE_module(128)


        self.MFA1 = MDSFA_block(64, 64, 0,num_heads, sr_ratio[0], upsample[0])
        self.MFA2 = MDSFA_block(128, 64, 1, num_heads, sr_ratio[1], upsample[1])
        self.MFA3 = MDSFA_block(256, 128, 1,num_heads, sr_ratio[2], upsample[2])
        self.MEE = MEE_module(128)

    def forward(self,x):#([2, 3, 224, 224])
               
        T_block1 = self.T_block1(x) #([2, 64, 56, 56])
        T_layer1 = self.T_layer1(T_block1) #([2, 64, 56, 56])
        T_layer1 = self.MFA1(T_layer1, T_block1)


        T_layer2 = self.T_layer2(T_layer1) #([2, 128, 28, 28])
        T_layer2 = self.MFA2(T_layer2, T_layer1)
        T_layer2 = self.MEE(T_layer2) #torch.Size([6,  128, 28, 28])

        T_layer3 = self.T_layer3(T_layer2) #([6, 256, 14, 14])
        T_layer3 = self.MFA3(T_layer3, T_layer2)

        # T_layer3 = self.DEE(T_layer3) #torch.Size([6, 256, 14, 14])

        #执行完DEE后通道数会从2变成6，因此需要使用repeat函数将x、T_layer1、T_layer2的通道数也变成6

        T_layer4 = self.T_layer4(T_layer3) #([6, 512, 7, 7])

        x =  x.repeat(3, 1, 1, 1)#torch.Size([6, 3, 224, 224])
        T_layer1 =  T_layer1.repeat(3, 1, 1, 1)#torch.Size([6, 64, 56, 56])
        # T_layer2 =  T_layer2.repeat(3, 1, 1, 1)#torch.Size([6, 128, 28, 28])
        
        #执行完DEE后通道数会从2变成6，因此需要使用repeat函数将x、T_layer1、T_layer2的通道数也变成6

        targets = [x.detach(),T_layer1.detach(),T_layer2.detach(),T_layer3.detach()]
        # ([6, 3, 224, 224])；([6, 64, 56, 56])；([6, 128, 28, 28])；([6, 256, 14, 14])
        ## 参数不更新--修改
        T_layer4 = T_layer4.detach() 


        S_layer3 = self.S_layer1(T_layer4) #([6, 256, 14, 14])
        S_layer2 = self.S_layer2(S_layer3) #([6, 128, 28, 28])
        S_layer1 = self.S_layer3(S_layer2) #([6, 64, 56, 56])
        S_layer0 = self.S_layer4(S_layer1) #torch.Size([6, 3, 224, 224])
        preds = [S_layer0,S_layer1,S_layer2,S_layer3]
        # ([6, 3, 224, 224]);([6, 64, 56, 56]);([6, 128, 28, 28]);([6, 256, 14, 14])
        outputs = [preds,targets]
        return outputs

if __name__ == "__main__":
    model = AE_T_r18_S_r18_()
    inputs = torch.zeros((3,224,224))
    outputs = model(inputs)
    print(outputs)
