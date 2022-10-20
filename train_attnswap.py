from data.data_loader_Swapping import tensor2im
from data.data_loader_Swapping import GetLoader
from models.model import fsModel
from util.plot import plot_batch
from util import util
import torch.utils.tensorboard as tensorboard
from torch.backends import cudnn
import torch.nn.functional as F
import torch
import os
from sched import scheduler
import sys
import time
import random
import argparse
import numpy as np
from copy import deepcopy
from PIL import Image

from argparse import Namespace
from models.psp import pSp
from torchvision import transforms
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


warnings.filterwarnings("ignore")


def str2bool(v):
    return v.lower() in ('true')


class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default='attnswap_1',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', default='0')
        self.parser.add_argument('--checkpoints_dir', type=str,
                                 default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--isTrain', type=str2bool, default='True')

        # input/output sizes
        self.parser.add_argument(
            '--batchSize', type=int, default=8, help='input batch size')  # batch size

        # for displays
        self.parser.add_argument(
            '--use_tensorboard', type=str2bool, default='True')  # use tensorboard for visualization

        # for trainingss
        self.parser.add_argument(
            '--dataset', type=str, default="./data_psp/celeba-256", help='path to he face swapping dataset')  # path to the face swapping dataset
        self.parser.add_argument('--continue_train', type=str2bool,
                                 default='False', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='./checkpoints/simswap224_test',
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='10000',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument(
            '--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument(
            '--niter', type=int, default=10000, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=10000,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument(
            '--beta1', type=float, default=0.0, help='momentum term of adam')
        self.parser.add_argument(
            '--G_lr', type=float, default=0.001, help='initial learning rate for adam')
        self.parser.add_argument(
            '--D_lr', type=float, default=0.00004, help='initial learning rate for adam')
        self.parser.add_argument('--Gdeep', type=str2bool, default='False')

        # for discriminators
        self.parser.add_argument(
            '--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument(
            '--lambda_id', type=float, default=30.0, help='weight for id loss')
        self.parser.add_argument(
            '--lambda_rec', type=float, default=10.0, help='weight for reconstruction loss')

        self.parser.add_argument(
            "--Arc_path", type=str, default='./arcface_model/arcface_checkpoint.tar', help="run ONNX model via TRT")
        self.parser.add_argument(
            "--PSP_path", type=str, default='./psp_model/psp_ffhq_encode.pt', help='the pretrained psp model')
        self.parser.add_argument(
            "--total_step", type=int, default=1000000, help='total training step')
        self.parser.add_argument(
            "--log_frep", type=int, default=20, help='frequence for printing log information')
        self.parser.add_argument(
            "--sample_freq", type=int, default=1000, help='frequence for sampling')
        self.parser.add_argument(
            "--model_freq", type=int, default=10000, help='frequence for saving the model')
        self.parser.add_argument(
            "--vgg_path", type=str, default='./vgg_model/vgg_normalised.pth', help="path of vgg model")
        self.parser.add_argument(
            "--vggface_path", type=str, default='./vggface_model/resnet50_ft_weight.pkl', help="path of vggface model")  # vggface的预训练权重

        self.isTrain = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        if self.opt.isTrain:
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            util.mkdirs(expr_dir)  # 创建保存训练log的文件夹
            if save and not self.opt.continue_train:
                file_name = os.path.join(expr_dir, 'opt.txt')  # 保存训练参数
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))  # 保存训练参数
                    opt_file.write('-------------- End ----------------\n')
        return self.opt


def lr_func(num_epochs):
    def func(s):
        if s < (num_epochs//2):
            return 1
        else:
            return max(0, 1-(2*s-num_epochs)/num_epochs)
    return func


if __name__ == '__main__':

    opt = TrainOptions().parse()
    iter_path = os.path.join(
        opt.checkpoints_dir, opt.name, 'iter.txt')  # 保存训练的迭代次数

    sample_path = os.path.join(
        opt.checkpoints_dir, opt.name, 'samples')  # 保存训练的样本
    os.makedirs(sample_path, exist_ok=True)

    log_path = os.path.join(opt.checkpoints_dir,
                            opt.name, 'summary')  # 保存训练的log
    os.makedirs(log_path, exist_ok=True)

    if opt.continue_train:  # 如果继续训练，就加载最新的模型的迭代次数
        try:
            start_epoch, epoch_iter = np.loadtxt(
                iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' %
              (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)  # 设置GPU
    print("GPU used : ", str(opt.gpu_ids))

    cudnn.benchmark = True
    model = fsModel()
    model.initialize(opt)

    #####################################################
    if opt.use_tensorboard:
        tensorboard_writer = tensorboard.SummaryWriter(log_path)  # tensorboard
        logger = tensorboard_writer

    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

    with open(log_name, "a") as log_file:
        now = time.strftime("%c")  # 获取当前时间
        log_file.write(
            '================ Training Loss (%s) ================\n' % now)

    # Generator和Discriminator的优化器
    optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D

    loss_avg = 0
    refresh_count = 0

    train_loader = GetLoader(opt.dataset, opt.batchSize, 8, 1234)  # 加载数据集

    randindex = [i for i in range(opt.batchSize)]
    random.shuffle(randindex)  # 打乱数据集

    if not opt.continue_train:
        start = 0
    else:
        start = int(opt.which_epoch)
    total_step = opt.total_step  # 总共训练多少步
    import datetime
    print("Start to train at %s" %
          (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    model.netD.feature_network.requires_grad_(False)  # 不更新feature network的参数

    # Training Cycle
    for step in range(start, total_step):
        model.netG.net.eval()
        model.netG.sanet1.train()
        model.netG.sanet2.train()
        model.netG.sanet3.train()
        model.netG.net.requires_grad_(False)
        for interval in range(2):
            random.shuffle(randindex)  # 打乱一个batch的数据 batch=16
            # 加载数据 shape: [batch, 3, 224, 224]
            src_image = train_loader.next()  # src_image1为原图，src_image2为风格图
            src_image1, src_image2 = deepcopy(src_image), deepcopy(src_image)

            if step % 2 == 0:
                img_id = src_image2  # 偶数次训练，重建原图
            else:
                img_id = src_image2[randindex]  # 奇数次训练，打乱数据集

            if interval:
                # img_fake为换脸后的图片 [batch, 3, 224, 224]
                img_fake = model.netG(src_image1, img_id)
                # gen_logits为生成器生成的图片的判别结果
                gen_logits, _ = model.netD(img_fake.detach(), None)

                loss_Dgen = (F.relu(torch.ones_like(
                    gen_logits) + gen_logits)).mean()  # loss_Dgen为生成器生成的图片的判别结果的loss， 为了让生成器生成的图片被判别器判别为真， loss_Dgen应该越小越好

                real_logits, _ = model.netD(
                    src_image2, None)  # real_logits为真实图片的判别结果
                loss_Dreal = (F.relu(torch.ones_like(
                    real_logits) - real_logits)).mean()  # loss_Dreal为真实图片的判别结果的loss， 其中真实图片的判别结果应该为1， 所以loss_Dreal为1-real_logits

                # loss_D为判别器的loss， 为了让生成器生成的图片被判别器判别为真， loss_D应该越小越好
                loss_D = loss_Dgen + loss_Dreal
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
            else:
                model.netD.requires_grad_(True)  # 更新判别器的参数
                img_fake = model.netG(src_image1, img_id)

                # G loss
                # gen_logits为生成器生成的图片的判别结果， feat为生成器生成的图片的特征
                gen_logits, feat = model.netD(img_fake, None)

                loss_Gmain = (-gen_logits).mean()

                loss_G_ID, _, _ = model.id_loss(img_fake, img_id, src_image1)
                if step % 2 == 1:
                    logger.add_scalar('loss_G_ID', loss_G_ID, step)  # ID loss

                real_feat = model.netD.get_feature(
                    src_image1)  # real_feat为真实图片的特征
                feat_match_loss = model.criterionFeat(
                    feat["3"], real_feat["3"])  # 生成人脸特征和原人脸特征的l1 loss
                logger.add_scalar('feat_match_loss',
                                  feat_match_loss, step)  # 特征匹配损失

                loss_G = loss_Gmain + loss_G_ID * opt.lambda_id + \
                    feat_match_loss * opt.lambda_feat

                if step % 2 == 0:
                    # G_Rec
                    loss_G_Rec = model.criterionRec(
                        img_fake, src_image1) * opt.lambda_rec
                    logger.add_scalar('loss_G_Rec', loss_G_Rec, step)  # 重建损失
                    loss_G += loss_G_Rec
                logger.add_scalar('loss_G', loss_G, step)

                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

        ############## Display results and errors ##########
        # print out errors
        # Print out log info
        if (step+1) % opt.log_frep == 0:
            # errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            errors = {
                "G_Loss": loss_Gmain.item(),
                "G_ID": loss_G_ID.item(),
                "G_Rec": loss_G_Rec.item(),
                "G_feat_match": feat_match_loss.item(),
                "D_fake": loss_Dgen.item(),
                "D_real": loss_Dreal.item(),
                "D_loss": loss_D.item()
            }
            if opt.use_tensorboard:
                for tag, value in errors.items():
                    logger.add_scalar(tag, value, step)
            message = '( step: %d, ) ' % (step)
            for k, v in errors.items():
                message += '%s: %.3f ' % (k, v)

            print(message)
            # print(model.netG.transform.sanet.gamma)
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)

        # display output images
        if (step+1) % opt.sample_freq == 0:
            model.netG.eval()
            with torch.no_grad():
                imgs = list()
                zero_img = (torch.zeros_like(src_image1[0, ...]))
                imgs.append(zero_img.cpu().numpy())
                save_img = src_image1
                save_img2 = src_image2
                for r in range(opt.batchSize):
                    imgs.append(tensor2im(save_img[r, ...]))

                for i in range(opt.batchSize):

                    imgs.append(tensor2im(save_img[i, ...]))
                    image_infer = src_image1[i, ...].repeat(
                        opt.batchSize, 1, 1, 1)
                    # Content, Style=model.Encoder(image_infer, src_image2)
                    # img_fake        = model.netG(Content, Style).cpu()

                    img_fake = model.netG(image_infer, src_image2)

                    for j in range(opt.batchSize):
                        imgs.append(tensor2im(img_fake[j, ...]))
                # print("Save test data")
                imgs = np.stack(imgs, axis=0).transpose(0, 2, 3, 1)
                print("Save test data which has {} images".format(
                    imgs.shape[0]))

                plot_batch(imgs, os.path.join(
                    sample_path, 'step_'+str(step+1)+'.jpg'))

        # save latest model
        if (step+1) % opt.model_freq == 0:
            print('saving the latest model (steps %d)' % (step+1))
            model.save(step+1)
            np.savetxt(iter_path, (step+1, total_step),
                       delimiter=',', fmt='%d')
