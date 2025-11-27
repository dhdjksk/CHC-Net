import torch
import torch.nn as nn
from ..modules.conv import Conv
from ..modules.block import *

__all__ = ["CSPP", "AFCF"]

class CSPP(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Pyramid(self.c) for _ in range(n))

class Pyramid(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()

        self.conv1 = Conv(inc, inc, k=3)
        self.conv2 = Conv(inc//2, inc//2, k=3)
        self.conv3 = Conv(inc//4, inc//4, k=3)
        self.conv4 = Conv(inc//8, inc//8, k=3)
        self.conv5 = Conv(inc, inc, 1)

        self.conv6 = Conv(inc // 16, inc // 16, k=3)
    def forward(self, x):
        conv1_out = self.conv1(x)
        conv1_out_1, conv1_out_2 = conv1_out.chunk(2, dim=1)
        conv2_out = self.conv2(conv1_out_1)
        conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)
        conv3_out = self.conv3(conv2_out_1)
        conv3_out_1, conv3_out_2 = conv3_out.chunk(2, dim=1)
        conv4_out = self.conv4(conv3_out_1)
        out = torch.cat([conv4_out, conv3_out_2, conv2_out_2, conv1_out_2], dim=1)
        out = self.conv5(out) + x
        return out


class AFCF(nn.Module):
    def __init__(self, inc, input_dim=64):
        super().__init__()

        self.input_dim = input_dim

        self.d_in1 = Conv(input_dim//2, input_dim//2, 1)
        self.d_in2 = Conv(input_dim//2, input_dim//2, 1)

        self.conv = Conv(input_dim, input_dim, 3)
        self.fc1 = nn.Conv2d(inc[1], input_dim//2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(inc[0], input_dim//2, kernel_size=1, bias=False)

        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        H_feature, L_feature = x

        H_feature = Upsample(H_feature, size=L_feature.size()[2:])

        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)

        g_L_feature = self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)

        L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * (g_H_feature * H_feature)
        H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * (g_L_feature * L_feature)

        out = self.conv(torch.cat([H_feature, L_feature], dim=1))

        return out

def Upsample(x, size, align_corners = False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='nearest')

