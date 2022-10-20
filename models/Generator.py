import torch.nn.functional as F
import math
import torch.nn as nn
import torch
import sys
import os
from argparse import Namespace
from models.psp import pSp
from models.stylegan2.model import Generator
from models.encoders import psp_encoders
dir = os.path.abspath(os.path.dirname(__file__))
Attn_dir = os.path.abspath(os.path.join(dir, '../'))
sys.path.append(dir)
sys.path.append(os.path.join(dir, 'encoders'))
sys.path.append(Attn_dir)
sys.path.append(os.path.abspath(os.path.join(dir, 'util')))
sys.path.append(os.path.abspath(os.path.join(dir, 'data')))

from encoders.psp_encoders import GradualStyleEncoder
from stylegan2.model import Generator

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items()
              if k[:len(name)] == name}
    return d_filt


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


encoder = nn.Sequential(
    nn.ReflectionPad2d(3),
    nn.Conv2d(3, 64, kernel_size=7, padding=0),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
)

decoder = nn.Sequential(
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
    nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
    nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ReflectionPad2d(3),
    nn.Conv2d(64, 3, kernel_size=7, padding=0),
)


class SANet(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_planes, activation=nn.ReLU()):
        super(SANet, self).__init__()
        self.chanel_in = in_planes
        self.activation = activation

        self.query_conv = nn.Conv2d(
            in_channels=in_planes, out_channels=in_planes//4, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_planes, out_channels=in_planes//4, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_planes, out_channels=in_planes, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, content, style):
        """
            inputs :
                x : input feature maps(BCWH)
            returns :
                out : self attention value + input feature 
                attention: BNN (N is Width*Height)
        """
        m_batchsize, C, width, height = content.size()
        proj_query = self.query_conv(content).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # BCN
        proj_key = self.key_conv(style).view(
            m_batchsize, -1, width*height)  # BC(*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)/(C**0.5)  # B(N)(N)
        proj_value = self.value_conv(style).view(
            m_batchsize, -1, width*height)  # BCN

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)  # BCWH

        out = self.gamma*out + content
        return out


class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet = SANet(in_planes=in_planes)
        self.merge_conv_pad = nn.ReflectionPad2d(
            (1, 1, 1, 1))  # 这里的padding是为了保证输入输出的size一致
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, Content, Style):
        return self.sanet(Content, Style)


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ResBlock(nn.Module):
    """
    Initialize a residual block with two convolutions followed by batchnorm layers
    """

    def __init__(self, in_size: int, hidden_size: int, out_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, out_size, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.batchnorm2 = nn.BatchNorm2d(out_size)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x
    """
    Combine output with the original input
    """

    def forward(self, x): return x + self.convblock(x)  # skip connection


class Generator_AttnSwap(nn.Module):
    def __init__(self):
        super(Generator_AttnSwap, self).__init__()
        ckpt = torch.load('./psp_model/psp_ffhq_encode.pt', map_location='cpu')
        print('load psp model success')
        opts = ckpt['opts']
        opts['checkpoint_path'] = './psp_model/psp_ffhq_encode.pt'
        opts['output_size'] = 1024
        opts['learn_in_w'] = False
        opts = Namespace(**opts)
        opts.n_styles = int(math.log(opts.output_size, 2)) * 2 - 2
        # PSP encoder
        self.net = pSp(opts)
        self.net.cuda()
        # SANet model
        self.sanet1 = SANet(in_planes=128)
        self.sanet2 = SANet(in_planes=256)
        self.sanet3 = SANet(in_planes=512)
        self.sanet1.cuda()
        self.sanet2.cuda()
        self.sanet3.cuda()

    def forward(self, tgt_img, src_img):
        tgt_feature_map1, tgt_feature_map2, tgt_feature_map3 = self.net(x=tgt_img)
        src_feature_map1, src_feature_map2, src_feature_map3 = self.net(x=src_img)
        c1 = self.sanet1(tgt_feature_map1, src_feature_map1)
        c2 = self.sanet2(tgt_feature_map2, src_feature_map2)
        c3 = self.sanet3(tgt_feature_map3, src_feature_map3)
        return self.net(x=None, c1=c1, c2=c2, c3=c3)
