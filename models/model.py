import torch
import torch.nn as nn
import torch

from models.base_model import BaseModel
from models.Generator import Generator_AttnSwap
from models.Discriminator import Discriminator_AttnSwap
from argparse import Namespace
from models.psp import pSp
from models.id_loss import IDLoss


def compute_grad2(d_out, x_in): 
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in, 
        create_graph=True, retain_graph=True, only_inputs=True
    )[0] # 计算梯度
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


class fsModel(BaseModel):
    def name(self):
        return 'fsModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # Generator network 生成器
        self.netG = Generator_AttnSwap()
        self.netG.cuda()
        self.netG.net.requires_grad_(False)

        # Id network 身份信息提取网络
        self.id_loss = IDLoss().cuda().eval()
        
        self.netD = Discriminator_AttnSwap(
            diffaug=False, interp224=False, **{})
        # self.netD.feature_network.requires_grad_(False)
        self.netD.cuda()

        if self.isTrain:
            # define loss functions
            self.criterionFeat = nn.L1Loss()
            self.criterionRec = nn.L1Loss()

           # initialize optimizers
            # optimizer G
            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(
                params, lr=opt.G_lr, betas=(opt.beta1, 0.99), eps=1e-8)

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(
                params, lr=opt.D_lr, betas=(opt.beta1, 0.99), eps=1e-8)

        # load networks
        if opt.continue_train:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            # print (pretrained_path)
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_G, 'G',
                            opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_D, 'D',
                            opt.which_epoch, pretrained_path)
        torch.cuda.empty_cache()

    def cosin_metric(self, x1, x2):
        # return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))  
        return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch)
        self.save_network(self.netD, 'D', which_epoch)
        self.save_optim(self.optimizer_G, 'G', which_epoch)
        self.save_optim(self.optimizer_D, 'D', which_epoch)
        '''if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)'''

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(
            params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
