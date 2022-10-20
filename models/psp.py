"""
This file defines the core research contribution
"""
from models.stylegan2.model import Generator
from models.encoders import psp_encoders
from torch import nn
import torch
import math
import matplotlib
matplotlib.use('Agg')


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items()
              if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):
    def __init__(self, opts):
        super(pSp, self).__init__()
        self.set_opts(opts)
        # compute number of style inputs based on the output resolution
        self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        self.encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        self.decoder = Generator(self.opts.output_size, 512, 8)  # 解码器
        self.face_pool = torch.nn.AdaptiveAvgPool2d(
            (256, 256))  # 将图片resize到256*256
        self.load_weights()

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading pSp from checkpoint: {}'.format(
                self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(
                get_keys(ckpt, 'encoder'), strict=True)
            self.decoder.load_state_dict(
                get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)

    def forward(self, x=None, c1=None, c2=None, c3=None, resize=True, latent_mask=None, input_code=False, randomize_noise=False,
                inject_latent=None, return_latents=False, alpha=None):
        if x is not None:
            map1, map2, map3 = self.encoder(x=x, c1=None, c2=None, c3=None)
            return map1, map2, map3
        else:
            codes = self.encoder(x=None, c1=c1, c2=c2, c3=c3)  # 输出的特征向量
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if self.opts.learn_in_w:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
                else:
                    codes = codes + \
                        self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + \
                            (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = not input_code
        images, result_latent = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None
