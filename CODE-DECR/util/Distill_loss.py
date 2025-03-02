from turtle import forward
from torch import nn
import torch

class mse_cosSimilar(nn.Module):
    def __init__(self) -> None:
        super(mse_cosSimilar,self).__init__()
    def forward(self,pred,target):
        mse_loss = torch.nn.MSELoss()
        cos_loss = torch.nn.CosineSimilarity()
        loss = 0
        for item in range(len(pred)):
            #print(a[item].shape)
            #print(b[item].shape)
            loss += 0.3*mse_loss(pred[item], target[item])
            loss += 0.7*torch.mean(1-cos_loss(pred[item].reshape(pred[item].shape[0],-1),
                                        target[item].reshape(target[item].shape[0],-1)))
        return loss


class cosSimilar(nn.Module):
    def __init__(self) -> None:
        super(cosSimilar,self).__init__()
    def forward(self,pred,target):
        cos_loss = torch.nn.CosineSimilarity()
        loss = 0
        for item in range(len(pred)):
            #print(a[item].shape)
            #print(b[item].shape)
            loss += torch.mean(1-cos_loss(pred[item].reshape(pred[item].shape[0],-1),
                                        target[item].reshape(target[item].shape[0],-1)))
            # loss += torch.mean(1-cos_loss(pred[item],target[item]))
        return loss

class mse(nn.Module):
    def __init__(self) -> None:
        super(mse,self).__init__()
    def forward(self,pred,target):
        mse_loss = torch.nn.MSELoss()
        loss = 0
        for item in range(len(pred)):
            #print(a[item].shape)
            #print(b[item].shape)
            loss += mse_loss(pred[item], target[item])
        return loss
