import math
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
from stylegan2.model import EqualLinear
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
        # H = (H_in - kernel_size + 2 * padding + stride) / stride = (H_in - 3 + 2 * 1 + 2) / 2 = H_in / 2
        # W = (W_in - kernel_size + 2 * padding + stride) / stride = (W_in - 3 + 2 * 1 + 2) / 2 = W_in / 2
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
    """
    The mapping network, map2style, is a small fully convolutional network, 
    which gradually reduces spatial size using a set of 2-strided convolutions followed by LeakyReLU activations. 
    Each generated 512 vector, is fed into StyleGAN, starting from its matching afﬁne transformation, A.
    """

    def __init__(self, num_layers, mode='ir', opts=None):
        self.input_nc = 3  # 输入的通道数
        self.output_size = 1024  # 输出图像的resolution大小
        self.n_styles = int(math.log(self.output_size, 2)) * \
            2 - 2  # 根据生成图像的分辨率确定生成的特征向量的个数
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
        self.style_count = self.n_styles  # 特征向量的个数
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

    def forward(self, x=None, c1=None, c2=None, c3=None):
        latents = []
        if x is not None: # 如果输入的是图像，则返回三个特征图
            x = self.input_layer(x)  # 输入层

            # 1. Feature maps are ﬁrst extracted using a standard feature pyramid over a ResNet backbone
            modulelist = list(self.body._modules.values())
            # 2. For each of the 18 target styles, a small mapping network is trained to extract the learned styles from the corresponding feature map
            for i, l in enumerate(modulelist):
                x = l(x)
                if i == 6:
                    c1 = x  # 保存第一个特征图
                elif i == 20:
                    c2 = x  # 保存第二个特征图
                elif i == 23:
                    c3 = x  # 保存第三个特征图

            return c1, c2, c3

        else: # 如果输入的是特征图，则返回风格编码器的输出
            # styles (0-2) are generated from the small feature map
            for j in range(self.coarse_ind):
                latents.append(self.styles[j](c3))  # 将第三个特征图输入到前三个风格编码器中

            # styles (3-6) are generated from the medium feature map
            p2 = self._upsample_add(c3, self.latlayer1(c2))
            for j in range(self.coarse_ind, self.middle_ind):
                latents.append(self.styles[j](p2))  # 将相加后的特征图输入到中间三个风格编码器中

            # styles (7-18) are generated from the largest feature map
            p1 = self._upsample_add(p2, self.latlayer2(c1))
            for j in range(self.middle_ind, self.style_count):
                latents.append(self.styles[j](p1))  # 将相加后的特征图输入到后面的风格编码器中

            out = torch.stack(latents, dim=1)  # 将风格编码器的输出拼接在一起
            return out
