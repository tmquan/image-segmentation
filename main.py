
"""
This example is largely adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py

Attributes:
    MODEL_NAMES (TYPE): Description
"""
import argparse
import os
import random
from pprint import pprint
from collections import OrderedDict

import numpy as np
import cv2 
import albumentations as AB

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

from tensorpack.dataflow import imgaug
from tensorpack.dataflow import AugmentImageComponent, AugmentImageComponents 
from tensorpack.dataflow import BatchData, MultiProcessRunner, PrintData, MapData, FixedSizeData

from data import CustomDataSet
from models.unet import UNet, UPPNet, FusionNet

class DiceLoss(nn.Module):
    def init(self):
        super(DiceLoss, self).init()

    def forward(self, output, target):
        smooth = 1.
        iflat = output.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

def DiceCoef(output, target):
    smooth = 1.
    iflat = output.ravel()
    tflat = target.ravel()
    intersection = (iflat * tflat).sum()
    A_sum = np.sum(iflat * iflat)
    B_sum = np.sum(tflat * tflat)
    return ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

class ImageNetLightningModel(LightningModule):
    """Summary

    Attributes:
        average_type (TYPE): Description
        criterion (TYPE): Description
        hparams (TYPE): Description
        model (TYPE): Description

    """
    def __init__(self, hparams):
        """
        TODO: add docstring here

        Args:
            hparams (TYPE): Description
        """
        super(ImageNetLightningModel, self).__init__()
        self.hparams = hparams
        if self.hparams.arch.lower() == 'unet':
            self.model = UNet(input_channels=1, num_classes=self.hparams.types)
        elif self.hparams.arch.lower() == 'uppnet':
            self.model = UPPNet(input_channels=1, num_classes=self.hparams.types)
        elif self.hparams.arch.lower() == 'fusionnet':
            self.model = FusionNet(input_channels=1, num_classes=self.hparams.types)
        else:
            ValueError
        # print(self.model.parameters())
        # logits -> nn.BCEWithLogitsLoss
        # logits -> sigmoid -> nn.BCELoss
        self.criterion = DiceLoss()

    def forward(self, x):
        """Summary

        Args:
            x (TYPE): Description

        Returns:
            TYPE: Description
        """
        y = (torch.tanh(self.model(x / 128.0 - 1.0)) / 2.0 + 0.5)*255.0
        return y

    def training_step(self, batch, batch_idx, prefix='train'):
        """Summary

        Args:
            batch (TYPE): Description
            batch_idx (TYPE): Description
            prefix (str, optional): Description

        Returns:
            TYPE: Description
        """
        images, target0, target1, target2, target3, target4, target5, = batch
        target = torch.cat([target0, target1, target2, target3, target4, target5], axis=1)

        output = self.forward(images)
        loss = self.criterion(output / 255.0, target / 255.0)

        tqdm_dict = {'train_loss': loss}
        result = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # viz = images
        # for p in range(self.hparams.types):
        #     viz = torch.cat([viz, target[:,p:p+1,:,:], output[:,p:p+1,:,:]], axis=-1)

        # self.logger.experiment.add_image(f'{prefix}_viz',
        #                                  torchvision.utils.make_grid(viz / 255.0, nrow=1,),
        #                                  dataformats='CHW', 
        #                                  global_step=self.global_step)

        return result

    def custom_step(self, batch, batch_idx, prefix='val'):
        """Summary

        Args:
            batch (TYPE): Description
            batch_idx (TYPE): Description
            prefix (str, optional): Description

        Returns:
            TYPE: Description
        """
        images, target0, target1, target2, target3, target4, target5, = batch
        target = torch.cat([target0, target1, target2, target3, target4, target5], axis=1)

        output = self.forward(images)
        loss = self.criterion(output / 255.0, target / 255.0)

        result = OrderedDict({
            f'{prefix}_loss': loss,
            f'{prefix}_output': output,
            f'{prefix}_target': target,
        })
        viz = images
        for p in range(self.hparams.types):
            # print(viz.shape, target.shape)
            viz = torch.cat([viz, target[:,p:p+1,:,:], output[:,p:p+1,:,:]], axis=-1)

        self.logger.experiment.add_image(f'{prefix}_viz',
                                         torchvision.utils.make_grid(viz / 255.0, nrow=1,),
                                         dataformats='CHW', 
                                         global_step=self.global_step)
        # viz = torchvision.utils.make_grid(images / 255.0)
        # # print(viz.shape)
        # viz = torch.cat([viz, torchvision.utils.make_grid(target[0,:,:] / 255.0)], axis=1)
        # viz = torch.cat([viz, torchvision.utils.make_grid(output[0,:,:] / 255.0)], axis=1)
        # self.logger.experiment.add_image(f'{prefix}_viz', viz, dataformats='CHW') 
        # self.logger.experiment.add_image(f'{prefix}_images',
        #                                  torchvision.utils.make_grid(images / 255.0),
        #                                  dataformats='CHW')
        # self.logger.experiment.add_image(f'{prefix}_output_0',
        #                                  torchvision.utils.make_grid(output[:,0] / 255.0),
        #                                  dataformats='CHW')
        # self.logger.experiment.add_image(f'{prefix}_target_0',
        #                                  torchvision.utils.make_grid(target[:,0] / 255.0),
        #                                  dataformats='CHW')
        return result

    def validation_step(self, batch, batch_idx, prefix='val'):
        """Summary

        Args:
            batch (TYPE): Description
            batch_idx (TYPE): Description
            prefix (str, optional): Description

        Returns:
            TYPE: Description
        """
        return self.custom_step(batch, batch_idx, prefix=prefix)

    def test_step(self, batch, batch_idx, prefix='test'):
        """Summary

        Args:
            batch (TYPE): Description
            batch_idx (TYPE): Description
            prefix (str, optional): Description

        Returns:
            TYPE: Description
        """
        return self.custom_step(batch, batch_idx, prefix=prefix)

    def custom_epoch_end(self, outputs, prefix='custom'):
        """Summary

        Args:
            outputs (TYPE): Description
            prefix (str, optional): Description

        Returns:
            TYPE: Description
        """
        loss_mean = torch.stack([x[f'{prefix}_loss'] for x in outputs]).mean()

        np_output = torch.cat([x[f'{prefix}_output'].squeeze_(0) for x in outputs], axis=0).to('cpu').numpy()
        np_target = torch.cat([x[f'{prefix}_target'].squeeze_(0) for x in outputs], axis=0).to('cpu').numpy()

        # Casting to binary
        result = {}
        result[f'{prefix}_loss'] = loss_mean

        tqdm_dict = {}
        tqdm_dict[f'{prefix}_loss'] = loss_mean

        tb_log = {}
        tb_log[f'{prefix}_loss'] = loss_mean

        dice_scores = []
        if np_output.shape[0] > 0 and np_target.shape[0] > 0:
            for p in range(self.hparams.types):
                dice_score = DiceCoef(np_output[:, p] / 255.0, np_target[:, p] / 255.0)
                tqdm_dict[f'{prefix}_dice_score_{p}'] = f'{dice_score:0.4f}'  
                tb_log[f'{prefix}_dice_score_{p}'] = dice_score     
                dice_scores.append(dice_score)      

        tqdm_dict[f'{prefix}_dice_score_mean'] = np.array(dice_scores).mean()
        tb_log[f'{prefix}_dice_score_mean'] = np.array(dice_scores).mean()
        
        pprint(tqdm_dict)
        result['log'] = tb_log

        return result

    def validation_epoch_end(self, outputs, prefix='val'):
        """Summary

        Args:
            outputs (TYPE): Description
            prefix (str, optional): Description

        Returns:
            TYPE: Description
        """
        return self.custom_epoch_end(outputs, prefix=prefix)

    def test_epoch_end(self, outputs, prefix='test'):
        """Summary

        Args:
            outputs (TYPE): Description
            prefix (str, optional): Description

        Returns:
            TYPE: Description
        """
        return self.custom_epoch_end(outputs, prefix=prefix)

    def configure_optimizers(self):
        """Summary

        Returns:
            TYPE: Description
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )
        return [optimizer], []

    @pl.data_loader
    def train_dataloader(self):
        """Summary

        Returns:
            TYPE: Description
        """
        ds_train = CustomDataSet(folder=self.hparams.data,
            train_or_valid='train',
            size=10000,
            hparams=self.hparams
            )
        ds_train.reset_state()
        ag_train = [
            imgaug.RotationAndCropValid(max_deg=10, interp=cv2.INTER_NEAREST),
            imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.8, 1.0), 
                    aspect_ratio_range=(0.8, 1.2),
                    interp=cv2.INTER_NEAREST, 
                    target_shape=self.hparams.shape),
            imgaug.Resize(self.hparams.shape, interp=cv2.INTER_NEAREST),
            imgaug.ToFloat32(),
        ]
        ds_train = AugmentImageComponent(ds_train, [
                                                    # imgaug.Float32(),
                                                    # imgaug.RandomChooseAug([
                                                    #     imgaug.Albumentations(AB.IAAAdditiveGaussianNoise(p=0.25)),
                                                    #     imgaug.Albumentations(AB.GaussNoise(p=0.25)),
                                                    #     ]), 
                                                    # imgaug.ToUint8(),
                                                    imgaug.RandomChooseAug([
                                                        imgaug.Albumentations(AB.Blur(blur_limit=4, p=0.25)),
                                                        imgaug.Albumentations(AB.MotionBlur(blur_limit=4, p=0.25)),
                                                        imgaug.Albumentations(AB.MedianBlur(blur_limit=4, p=0.25)),
                                                        ]),
                                                    imgaug.RandomChooseAug([
                                                        # imgaug.Albumentations(AB.IAASharpen(p=0.5)),
                                                        # imgaug.Albumentations(AB.IAAEmboss(p=0.5)),
                                                        imgaug.Albumentations(AB.RandomBrightnessContrast(p=0.5)),
                                                        ]),
                                                    imgaug.ToUint8(),
                                                    imgaug.Albumentations(AB.CLAHE(p=0.5)),
                                                        
                                                   ], 0)
        ds_train = AugmentImageComponents(ds_train, ag_train, [0, 1, 2, 3, 4, 5, 6])
        ds_train = PrintData(ds_train)
        
        ds_train = BatchData(ds_train, self.hparams.batch, remainder=True)
        if self.hparams.debug:
            ds_train = FixedSizeData(ds_train, 2)
        ds_train = MultiProcessRunner(ds_train, num_proc=4, num_prefetch=16)
        ds_train = PrintData(ds_train)
        ds_train = MapData(ds_train, lambda dp: [torch.tensor(dp[0][:,np.newaxis,:,:]).float(), 
                                                 torch.tensor(dp[1][:,np.newaxis,:,:]).float(),
                                                 torch.tensor(dp[2][:,np.newaxis,:,:]).float(),
                                                 torch.tensor(dp[3][:,np.newaxis,:,:]).float(),
                                                 torch.tensor(dp[4][:,np.newaxis,:,:]).float(),
                                                 torch.tensor(dp[5][:,np.newaxis,:,:]).float(),
                                                 torch.tensor(dp[6][:,np.newaxis,:,:]).float(),
                                                 ])
        return ds_train

    @pl.data_loader
    def val_dataloader(self):
        """Summary

        Returns:
            TYPE: Description
        """
        ds_valid = CustomDataSet(folder=self.hparams.data,
            train_or_valid='valid',
            size=10000,
            hparams=self.hparams
            )

        ds_valid.reset_state()
        ag_valid = [
            imgaug.Resize(self.hparams.shape, interp=cv2.INTER_NEAREST),
            imgaug.ToFloat32(),
        ]
        ds_valid = PrintData(ds_valid)
        ds_valid = AugmentImageComponent(ds_valid, [imgaug.Albumentations(AB.CLAHE(p=1)),
                                                    ], 0)
        ds_valid = AugmentImageComponents(ds_valid, ag_valid, [0, 1, 2, 3, 4, 5, 6])
        ds_valid = BatchData(ds_valid, self.hparams.batch, remainder=True)
        # ds_valid = MultiProcessRunner(ds_valid, num_proc=4, num_prefetch=16)
        ds_valid = MapData(ds_valid, lambda dp: [torch.tensor(dp[0][:,np.newaxis,:,:]).float(), 
                                                 torch.tensor(dp[1][:,np.newaxis,:,:]).float(), 
                                                 torch.tensor(dp[2][:,np.newaxis,:,:]).float(), 
                                                 torch.tensor(dp[3][:,np.newaxis,:,:]).float(), 
                                                 torch.tensor(dp[4][:,np.newaxis,:,:]).float(), 
                                                 torch.tensor(dp[5][:,np.newaxis,:,:]).float(), 
                                                 torch.tensor(dp[6][:,np.newaxis,:,:]).float(), 
                                                 ])
        return ds_valid

    @pl.data_loader
    def test_dataloader(self):
        """Summary

        Returns:
            TYPE: Description
        """
        ds_test = CustomDataSet(folder=self.hparams.data,
            train_or_valid='test',
            size=10000,
            hparams=self.hparams
            )

        ds_test.reset_state()
        ag_test = [
            imgaug.Resize(self.hparams.shape, interp=cv2.INTER_NEAREST),
            imgaug.ToFloat32(),
        ]
        ds_test = AugmentImageComponent(ds_test, [imgaug.Albumentations(AB.CLAHE(p=1)),
                                                  ], 0)
        ds_test = AugmentImageComponent(ds_test, ag_test, 0)
        ds_test = BatchData(ds_test, self.hparams.batch, remainder=True)
        # ds_test = MultiProcessRunner(ds_test, num_proc=4, num_prefetch=16)
        ds_test = PrintData(ds_test)
        ds_test = MapData(ds_test, lambda dp: [torch.tensor(dp[0][:,np.newaxis,:,:]).float(), 
                                               torch.tensor(dp[1][:,np.newaxis,:,:]).float()])
        return ds_test

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """Summary

        Args:
            parent_parser (TYPE): Description

        Returns:
            TYPE: Description
        """
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('-a', '--arch', metavar='ARCH', default='unet', 
                            help='model architecture')
        parser.add_argument('--epochs', default=250, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=2222,
                            help='seed for initializing training. ')
        parser.add_argument('-b', '--batch', default=32, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        parser.add_argument('--debug', action='store_true',
                            help='use fast mode')
        return parser

def get_args():
    """Summary

    Returns:
        TYPE: Description
    """
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--data', metavar='DIR', default=".", type=str,
                               help='path to dataset')
    parent_parser.add_argument('--save', metavar='DIR', default="train_log", type=str,
                               help='path to save output')
    parent_parser.add_argument('--info', metavar='DIR', default="train_log",
                               help='path to logging output')
    parent_parser.add_argument('--gpus', type=int, default=1,
                               help='how many gpus')
    parent_parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('--use-16bit', dest='use_16bit', action='store_true',
                               help='if true uses 16 bit precision')
    parent_parser.add_argument('--percent_check', default=1.0, type=float,
                               help="float/int. If float, % of tng epoch. If int, check every n batch")
    parent_parser.add_argument('--val_check_interval', default=1.0, type=float,
                               help="float/int. If float, % of tng epoch. If int, check every n batch")
    parent_parser.add_argument('--fast_dev_run', default=False, action='store_true',
                               help='fast_dev_run: runs 1 batch of train, test, val (ie: a unit test)')

    parent_parser.add_argument('--types', type=int, default=1)
    parent_parser.add_argument('--threshold', type=float, default=0.5)
    parent_parser.add_argument('--pathology', default='All')
    parent_parser.add_argument('--shape', type=int, default=256)
    # parent_parser.add_argument('--folds', type=int, default=5)

    # Inference purpose
    # parent_parser.add_argument('--load', help='load model')
    parent_parser.add_argument('--load', action='store_true',
                               help='path to logging output')
    parent_parser.add_argument('--pred', action='store_true',
                               help='run predict')
    parent_parser.add_argument('--eval', action='store_true',
                               help='run offline evaluation instead of training')

    parser = ImageNetLightningModel.add_model_specific_args(parent_parser)
    return parser.parse_args()


def main(hparams):
    """Summary

    Args:
        hparams (TYPE): Description
    """
    # for valid_fold_index in range(hparams.folds):
    #     hparams.valid_fold_index = valid_fold_index
    model = ImageNetLightningModel(hparams)
    if hparams.seed is not None:
        random.seed(hparams.seed)
        np.random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(str(hparams.save),
                              str(hparams.arch),
                              str(hparams.pathology),
                              str(hparams.shape),
                              str(hparams.types),
                              # str(hparams.folds),
                              # str(hparams.valid_fold_index),
                              'ckpt'),
        save_top_k=10,
        verbose=True,
        monitor='val_loss',  # TODO
        mode='min'
    )

    trainer = pl.Trainer(
        train_percent_check=hparams.percent_check,
        val_percent_check=hparams.percent_check,
        test_percent_check=hparams.percent_check,
        num_sanity_val_steps=0,
        default_save_path=os.path.join(str(hparams.save),
                                   str(hparams.arch),
                                   str(hparams.pathology),
                                   str(hparams.shape),
                                   str(hparams.types),
                                   # str(hparams.folds),
                                   # str(hparams.valid_fold_index)
                                   ),
        gpus=hparams.gpus,
        max_epochs=hparams.epochs,
        checkpoint_callback=checkpoint_callback,
        progress_bar_refresh_rate=1,
        early_stop_callback=None,
        distributed_backend=hparams.distributed_backend,
        use_amp=hparams.use_16bit,
        val_check_interval=hparams.val_check_interval,
    )
    if hparams.eval:
        trainer.run_evaluation()
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main(get_args())
