from stylegan2.model import EqualLinear
from encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
import math

import sys
import os
dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir)
sys.path.append(os.path.join(dir, 'encoders'))
sys.path.append(os.path.join(dir, 'mtcnn'))
sys.path.append(os.path.join(dir, 'stylegan2'))
Attn_dir = os.path.abspath(os.path.join(dir, '../'))
sys.path.append(Attn_dir)
sys.path.append(os.path.abspath(os.path.join(dir, 'util')))
sys.path.append(os.path.abspath(os.path.join(dir, 'data')))


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c  # 输出的通道数
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        self.input_nc = 3
        self.output_size = 512
        self.n_styles = int(math.log(self.output_size, 2)) * 2 - 2
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [
            50, 100, 152], 'num_layers should be 50,100, or 152'  # 表示使用的是resnet50，100，152
        # ir表示残差网络，ir_se表示残差网络+SE模块
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR  # 残差网络
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE  # 残差网络+SE模块
        self.input_layer = Sequential(Conv2d(self.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))  # 输入层, 输出通道数为64
        modules = []
        for block in blocks:  # 依次添加残差网络
            for bottleneck in block:  # bottleneck表示残差网络的一个单元
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))  # 添加残差网络
        self.body = Sequential(*modules)  # 将残差网络添加到Sequential中

        self.styles = nn.ModuleList()  # 用于存储风格编码器
        self.style_count = self.n_styles  # 风格编码器的个数
        self.coarse_ind = 3  # 粗编码器的位置
        self.middle_ind = 7  # 中编码器的位置
        for i in range(self.style_count):
            if i < self.coarse_ind:  # 前三个风格编码器
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:  # 中间三个风格编码器
                style = GradualStyleBlock(512, 512, 32)
            else:  # 后面的风格编码器
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)  # 将风格编码器添加到ModuleList中
        self.latlayer1 = nn.Conv2d(
            256, 512, kernel_size=1, stride=1, padding=0)  # 用于将256通道的特征图转换为512通道的特征图
        self.latlayer2 = nn.Conv2d(
            128, 512, kernel_size=1, stride=1, padding=0)  # 用于将128通道的特征图转换为512通道的特征图

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps. 
        Args:
          x: (Variable) top feature map to be upsampled. x为上采样的特征图
          y: (Variable) lateral feature map. y为侧向特征图
        Returns:
          (Variable) added feature map. 返回添加后的特征图
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()  # 获取侧向特征图的大小
        # 上采样后与侧向特征图相加
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)  # 输入层

        latents = []
        modulelist = list(self.body._modules.values())  # 获取body中的所有模块
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x  # 保存第一个侧向特征图
            elif i == 20:
                c2 = x  # 保存第二个侧向特征图
            elif i == 23:
                c3 = x  # 保存第三个侧向特征图

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))  # 将第三个侧向特征图输入到前三个风格编码器中

        p2 = self._upsample_add(c3, self.latlayer1(c2))  # 将第三个侧向特征图与第二个侧向特征图相加
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))  # 将相加后的特征图输入到中间三个风格编码器中

        p1 = self._upsample_add(p2, self.latlayer2(c1))  # 将相加后的特征图与第一个侧向特征图相加
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))  # 将相加后的特征图输入到后面的风格编码器中

        out = torch.stack(latents, dim=1)  # 将风格编码器的输出拼接在一起
        return out


# 输出网络输出的维度
print(GradualStyleEncoder(50, 'ir_se')(torch.randn(16, 3, 214, 214)).shape)
