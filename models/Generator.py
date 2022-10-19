import torch.nn.functional as F
import math
import torch.nn as nn
import torch
import sys
import os
dir = os.path.abspath(os.path.dirname(__file__))
Attn_dir = os.path.abspath(os.path.join(dir, '../'))
sys.path.append(dir)
sys.path.append(os.path.join(dir, 'encoders'))
sys.path.append(Attn_dir)
sys.path.append(os.path.abspath(os.path.join(dir, 'util')))
sys.path.append(os.path.abspath(os.path.join(dir, 'data')))

from encoders.psp_encoders import GradualStyleEncoder
from stylegan2.model import Generator

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
    def __init__(self, opt):
        super(Generator_AttnSwap, self).__init__()
        self.encoder_psp = GradualStyleEncoder(50, 'ir_se')  # psp encoder
        self.decoder_psp = Generator(512, 512, 8) # psp decoder
        self.encoder_p2p = encoder # p2p encoder
        self.decoder_p2p = decoder # p2p decoder
        self.transform = Transform(in_planes=512) # sanet transform -- mix style and content
        self.resblock = ResBlock(in_size=512, hidden_size=512, out_size=512)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224)) # 将图片resize到224*224
        self.resblock.cuda()
        self.transform.cuda()
        self.encoder_psp.cuda()
        self.decoder_psp.cuda()
        self.encoder_p2p.cuda()
        self.decoder_p2p.cuda()
        

    def forward(self, tgt_img, src_img, resize=True):
        # tgt_img src_img shape: (16 * 3 * 224 * 224)
        Content = self.encoder_psp(tgt_img)
        Style = self.encoder_psp(src_img)
        # # Content shape: (16 * 16 * 512)
        # Content = Content.permute(2, 0, 1).unsqueeze(0)
        # Style = Style.permute(2, 0, 1).unsqueeze(0)
        # # Content shape: (1 * 512 * 16 * 16)
        # Content = self.resblock(Content)
        # Style = self.resblock(Style)
        # # Content shape: (1 * 512 * 16 * 16)
        # out = self.transform(Content, Style) # sanet transform
        # # out shape: (1 * 512 * 16 * 16)
        # result = out.squeeze(0).permute(1, 2, 0)
        # # result shape: (16 * 16 * 512)
        result = Content + Style
        images, result_latent = self.decoder_psp(result)
        if resize:
            images = self.face_pool(images)
        # images shape: (16 * 3 * 224 * 224)
            
        return images
