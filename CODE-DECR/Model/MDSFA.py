import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_

class CNL(nn.Module):
    def __init__(self, high_dim, low_dim, flag=0,
                 num_heads=8, sr_ratio=1, upsample=None):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        qkv_bias = True
        dim = low_dim
        drop_rate = 0.
        self.attn_drop = nn.Dropout(drop_rate)
        self.scale = low_dim ** -0.5

        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        assert high_dim % num_heads == 0, f"dim {high_dim} should be divided by num_heads {num_heads}."
        assert low_dim % num_heads == 0, f"dim {low_dim} should be divided by num_heads {num_heads}."
        self.q = nn.Linear(high_dim, dim, bias=qkv_bias)
        self.num_heads = num_heads
        self.upsample = upsample
        # 8,4,2
        self.sr_ratio = sr_ratio

        # if sr_ratio > 1:
        self.act = nn.GELU()
        if sr_ratio == 16:
            self.sr1 = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=16, stride=16)
            self.norm1 = nn.LayerNorm(self.low_dim)
            self.sr2 = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=8, stride=8)
            self.norm2 = nn.LayerNorm(self.low_dim)
        if sr_ratio == 8:
            self.sr1 = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=8, stride=8)
            self.norm1 = nn.LayerNorm(self.low_dim)
            self.sr2 = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=4, stride=4)
            self.norm2 = nn.LayerNorm(self.low_dim)
        if sr_ratio == 4:
            self.sr1 = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=4, stride=4)
            self.norm1 = nn.LayerNorm(self.low_dim)
            self.sr2 = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=2, stride=2)
            self.norm2 = nn.LayerNorm(self.low_dim)
        if sr_ratio == 2:
            self.sr1 = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=2, stride=2)
            self.norm1 = nn.LayerNorm(self.low_dim)
            self.sr2 = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1)
            self.norm2 = nn.LayerNorm(self.low_dim)

        self.apply(self._init_weights)

        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
            self.W1 = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(high_dim), )
            self.W2 = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(high_dim), )
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=2, padding=0)
            self.W1 = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0),
                                    nn.BatchNorm2d(self.high_dim), )
            self.W2 = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0),
                                    nn.BatchNorm2d(self.high_dim), )
        nn.init.constant_(self.W1[1].weight, 0.0)
        nn.init.constant_(self.W1[1].bias, 0.0)
        nn.init.constant_(self.W2[1].weight, 0.0)
        nn.init.constant_(self.W2[1].bias, 0.0)
        self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.local_conv1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
        self.local_conv2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
        self.apply(self._init_weights)

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x_h, x_l):
        B, Ch, hh, wh = x_h.shape
        x_hH = x_h.reshape(B, self.high_dim, -1).transpose(-2, -1)  # B,Nh,Ch
        Nh = hh * wh

        bl, cl, hl, wl = x_l.shape
        x_ll = x_l.reshape(B, self.low_dim, -1).transpose(-2, -1)  # B, Nl, C
        _, N, C = x_ll.shape  # torch.Size([48, 64, 3456])
        # print('1'*100)
        # print(x_l.shape)
        q_yuan = self.q(x_hH).reshape(B, Nh, self.num_heads, C // self.num_heads).permute(0, 2, 1,3)  # B,head ,Nh,C//head
        g_x = self.g(x_l).view(B, self.low_dim, -1)#1*1卷积

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)#1*1卷积
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)#1*1卷积

        if self.sr_ratio > 1:
            x_ = x_l
            x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))  # B, Nl//16, C
            x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
            kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                    4)  # B, Nl//16, 2, head //2, C//head --> 2,B,h//2, Nl//16 , c//h
            kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k1, v1 = kv1[0], kv1[1]  # B head//2, Nl//16, C//h
            k2, v2 = kv2[0], kv2[1]
            attn1 = (q_yuan[:, :self.num_heads // 2] @ k1.transpose(-2, -1)) * self.scale  # B head//2, Nh , Nl//16
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)
            v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).
                                       transpose(1, 2).view(B, C // 2, hl // self.sr_ratio, wl // self.sr_ratio)). \
                view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
            x1 = (attn1 @ v1).transpose(1, 2).reshape(B, Nh,
                                                      C // 2)  # #  B head//2, Nh , C//h -->  B Nh ,head//2,  C//h  -- > B Nh , C//2
            attn2 = (q_yuan[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2).
                                       transpose(1, 2).view(B, C // 2, hl * 2 // self.sr_ratio,
                                                            wl * 2 // self.sr_ratio)). \
                view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
            x2 = (attn2 @ v2).transpose(1, 2).reshape(B, Nh,
                                                      C // 2)  # B head//2, Nh , C//h -->  B Nh ,head//2,  C//h  -- > B Nh , C//2

            y = torch.cat([x1, x2], dim=-1)  # B,Nh,Cl

        y1_ = y.transpose(1, 2).reshape(B, self.low_dim,
                                        *x_h.size()[
                                         2:])  # B, low_dim ,hl,wl;    B, head//2 ,2C//head, -1(h),  C= C(l) //2
        if self.upsample == True:
            y1_ = self.up(y1_)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)
        # print('-' * 100)
        # print(g_x.shape)  # 64
        # print(attention.shape)  # 64
        y1 = torch.matmul(attention, g_x)  # #  B, C,Nl
        # y1 = y1.permute(0, 2, 1).contiguous() # B, Nl, C
        y12 = y1.view(B, self.low_dim, *x_l.size()[2:])
        # y1___ = y1_ +y12

        W_y = self.W1(y1_)
        W_y2 = self.W2(y12)
        # print('-' * 100)
        # print(W_y.shape)  # 64
        # print(x_h.shape) # 96,36

        z = W_y + W_y2 + x_h
        
        return z

class PNL(nn.Module):
    def __init__(self, high_dim, low_dim, reduc_ratio=2,
                 num_heads=8, sr_ratio=1):
        super(PNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio

        self.scale = low_dim ** -0.5
        drop_rate = 0.
        self.attn_drop = nn.Dropout(drop_rate)
        dim = self.low_dim
        qkv_bias = True
        # --------------
        self.num_heads = num_heads
        self.q = nn.Linear(high_dim, self.low_dim, bias=qkv_bias)

        # 8,4,2
        self.sr_ratio = sr_ratio
        # if sr_ratio > 1:
        self.act = nn.GELU()
        if sr_ratio == 16:
            self.sr1 = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=16, stride=16)
            self.norm1 = nn.LayerNorm(self.low_dim)
            self.sr2 = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=8, stride=8)
            self.norm2 = nn.LayerNorm(self.low_dim)
        if sr_ratio == 8:
            self.sr1 = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=8, stride=8)
            self.norm1 = nn.LayerNorm(self.low_dim)
            self.sr2 = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=4, stride=4)
            self.norm2 = nn.LayerNorm(self.low_dim)
        if sr_ratio == 4:
            self.sr1 = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=4, stride=4)
            self.norm1 = nn.LayerNorm(self.low_dim)
            self.sr2 = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=2, stride=2)
            self.norm2 = nn.LayerNorm(self.low_dim)
        if sr_ratio == 2:
            self.sr1 = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=2, stride=2)
            self.norm1 = nn.LayerNorm(self.low_dim)
            self.sr2 = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1)
            self.norm2 = nn.LayerNorm(self.low_dim)
        self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.local_conv1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
        self.local_conv2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
        self.new_conv2 = nn.Sequential(nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1),
                                       nn.BatchNorm2d(dim // 2),
                                       nn.ReLU(inplace=True)
                                       )
        self.apply(self._init_weights)

        self.g = nn.Conv2d(self.low_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim//self.reduc_ratio, kernel_size=1, stride=1, padding=0)

        self.W1 = nn.Sequential(
            nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(high_dim), )
        self.W2 = nn.Sequential(
            nn.Conv2d(self.low_dim // self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(high_dim), )
        nn.init.constant_(self.W1[1].weight, 0.0)
        nn.init.constant_(self.W1[1].bias, 0.0)
        nn.init.constant_(self.W2[1].weight, 0.0)
        nn.init.constant_(self.W2[1].bias, 0.0)
        # self.kv1 = self.phi
        # self.kv2 = self.phi

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x_h, x_l):
        B, Ch, hh, wh = x_h.shape
        bl, cl, hl, wl = x_l.shape
        x_ll = x_l.reshape(B, self.low_dim, -1).transpose(-2, -1)
        x_hH = x_h.reshape(B, self.high_dim, -1).transpose(-2, -1)  # B,Nh,Ch
        N = hh * wh
        B, _, C = x_ll.shape
        q_yuan = self.q(x_hH).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                         3)  ## B,Nh,Cl -->B,Nh,head, Cl//head  -->  B,head ,Nh,Cl//head
        # x_lll = self.g(x_l)
        g_x = self.g(x_l).reshape(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x_h).reshape(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x_l).reshape(B, self.low_dim, -1)

        # --------------------
        if self.sr_ratio > 1:
            x_ = x_l
            x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
            x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
            kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
            kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k1, v1 = kv1[0], kv1[1]  # # B head//2, Nl//16, Cl//h
            k2, v2 = kv2[0], kv2[1]
            attn1 = (q_yuan[:, :self.num_heads // 2] @ k1.transpose(-2, -1)) * self.scale  # B head//2, Nh , Nl//16
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)
            v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).
                                       transpose(1, 2).view(B, C // 2, hl // self.sr_ratio, wl // self.sr_ratio)). \
                view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
            x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N,
                                                      C // 2)  # # B head//2, Nh , Cl//h  - >-->  B Nh ,head//2, Cl//h  -- > B Nh , Cl//2
            attn2 = (q_yuan[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2).
                                       transpose(1, 2).view(B, C // 2, hl * 2 // self.sr_ratio,
                                                            wl * 2 // self.sr_ratio)). \
                view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
            x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N,
                                                      C // 2)  # # B head//2, Nh , Cl//h  - >-->  B Nh ,head//2, Cl//h  -- > B Nh , Cl//2

            y = torch.cat([x1, x2], dim=-1)  # B Nh , Cl
            y_ = y.transpose(1, 2).reshape(B, self.low_dim, *x_h.size()[2:])
        # y = self.new_conv2(y__)

        # y_ = y_.transpose(1, 2).reshape(B, self.low_dim//self.reduc_ratio, *x_h.size()[2:])  # B, low_dim ,h,w;    B, head//2 ,2C//head, -1(h),  C= C(l) //2

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y1 = torch.matmul(attention, g_x)
        y1 = y1.permute(0, 2, 1).contiguous()
        y1 = y1.view(B, self.low_dim // self.reduc_ratio, *x_h.size()[2:])
        # print('1'*100)
        # print(y1.shape)

        W_y = self.W1(y_)
        W_y1 = self.W2(y1)
        # yzong = y_ + y1
        # print('-' * 100)
        # print(W_y.shape)  # 64
        # print(x_h.shape) # 96,36
        z = W_y + W_y1 + x_h

        return z


class MDSFA_block(nn.Module):
    def __init__(self, high_dim, low_dim, flag, num_heads,sr_ratio, upsample ):
        super(MDSFA_block, self).__init__()

        # self.CNL = CNL(high_dim, low_dim, flag)
        # self.PNL = PNL(high_dim, low_dim)

        self.CNL = CNL(high_dim, low_dim, flag, num_heads=num_heads, sr_ratio=sr_ratio, upsample=upsample)
        self.PNL = PNL(high_dim, low_dim, num_heads=num_heads, sr_ratio=sr_ratio)

    def forward(self, x, x0):
        z = self.CNL(x, x0)
        z = self.PNL(z, x0)
        return z
    
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

class MEE_module(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MEE_module, self).__init__()

        self.FC11 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC11.apply(weights_init_kaiming)
        self.FC12 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC12.apply(weights_init_kaiming)
        self.FC13 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC13.apply(weights_init_kaiming)
        self.FC1 = nn.Conv2d(channel//4, channel, kernel_size=1)
        self.FC1.apply(weights_init_kaiming)

        self.FC21 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC21.apply(weights_init_kaiming)
        self.FC22 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC22.apply(weights_init_kaiming)
        self.FC23 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC23.apply(weights_init_kaiming)
        self.FC2 = nn.Conv2d(channel//4, channel, kernel_size=1)
        self.FC2.apply(weights_init_kaiming)
        self.dropout = nn.Dropout(p=0.01)

    def forward(self, x):
        x1 = (self.FC11(x) + self.FC12(x) + self.FC13(x))/3 #torch.Size([2, 64, 14, 14])
        x1 = self.FC1(F.relu(x1)) #torch.Size([2, 256, 14, 14])
        x2 = (self.FC21(x) + self.FC22(x) + self.FC23(x))/3 #torch.Size([2, 64, 14, 14])
        x2 = self.FC2(F.relu(x2)) #torch.Size([2, 256, 14, 14])
        out = torch.cat((x, x1, x2), 0) #torch.Size([6, 256, 14, 14])
        out = self.dropout(out) #torch.Size([6, 256, 14, 14])
        return out
