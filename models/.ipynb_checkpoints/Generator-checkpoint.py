from turtle import forward
import torch
import torch.nn as nn
import math
from models.vgg_face import resnet50
from util.util import load_state_dict
import torch.nn.functional as F

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

# decoder = nn.Sequential(
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(512, 256, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 128, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 128, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 64, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 64, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 3, (3, 3)),
# )

#
encoder=nn.Sequential(
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

decoder=nn.Sequential(
    nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
    nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(256), 
    nn.ReLU(True),
    nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
    nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(128), 
    nn.ReLU(True),
    nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64), 
    nn.ReLU(True),
    nn.ReflectionPad2d(3), 
    nn.Conv2d(64, 3, kernel_size=7, padding=0),
)

# class SANet(nn.Module):
    
#     def __init__(self, in_planes):
#         super(SANet, self).__init__()
#         self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
#         self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
#         self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
#         self.sm = nn.Softmax(dim = -1)
#         self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))
        
#     def forward(self, content, style):
#         F = self.f(mean_variance_norm(content))
#         G = self.g(mean_variance_norm(style))
#         H = self.h(style)
#         b, c, h, w = F.size()
#         F = F.view(b, -1, w * h).permute(0, 2, 1)
#         b, c, h, w = G.size()
#         G = G.view(b, -1, w * h)
#         S = torch.bmm(F, G)
#         S = self.sm(S)
#         b, c, h, w = H.size()
#         H = H.view(b, -1, w * h)
#         O = torch.bmm(H, S.permute(0, 2, 1))
#         b, c, h, w = content.size()
#         O = O.view(b, c, h, w)
#         O = self.out_conv(O)
#         O += content
#         return O
    
    
class SANet(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_planes,activation=nn.ReLU()):
        super(SANet,self).__init__()
        self.chanel_in = in_planes
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_planes , out_channels = in_planes//4 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_planes , out_channels = in_planes//4, kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_planes , out_channels = in_planes , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,content, style):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = content.size()
        proj_query  = self.query_conv(content).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(style).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(style).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + content
        return out
    
    
class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet = SANet(in_planes = in_planes)
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
    def forward(self, Content, Style):
        # return self.merge_conv(self.merge_conv_pad(self.sanet(Content, Style)))
        return self.sanet(Content, Style)
    
def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))    

class ResBlock(nn.Module):
    """
    Initialize a residual block with two convolutions followed by batchnorm layers
    """
    def __init__(self, in_size:int, hidden_size:int, out_size:int):
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
    def forward(self, x): return x + self.convblock(x) # skip connection


class Generator_AttnSwap(nn.Module):
    def __init__(self):
        super(Generator_AttnSwap, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.transform = Transform(in_planes = 512)
        BN = []
        for i in range(4):
            BN += [ResBlock(512,512,512)]
        self.resblock = nn.Sequential(*BN)
        self.transform.cuda()
        self.encoder.cuda()
        self.decoder.cuda()
    def forward(self, tgt_img, src_img):
        Content=self.encoder(tgt_img)
        Style=self.encoder(src_img)
        out1=self.transform(Content, Style)
        out2=self.resblock(out1)
        content = self.decoder(out2)
        return content
        
        
        
# class Encoder(nn.Module):
#     def __init__(self,opt):
#         super(Encoder,self).__init__()
#         self.model= resnet50(num_classes=8631, include_top=True)
#         load_state_dict(self.model, opt.vggface_path)
#         for param in self.model.parameters():
#             param.requires_grad = False
#     def forward(self, tgt_img, src_img):# content=tgt_img, style=src_img
#         Content = self.model(tgt_img)
#         Style = self.model(src_img)
#         return Content, Style


