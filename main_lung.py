
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
import segmentation_models_pytorch as smp

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

from tensorpack.dataflow import imgaug
from tensorpack.dataflow import AugmentImageComponent, AugmentImageComponents 
from tensorpack.dataflow import BatchData, MultiProcessRunner, PrintData, MapData, FixedSizeData

# from data import CustomDataSet
from models.unet import UNet, UPPNet, FusionNet

import tensorpack.dataflow 
from tensorpack.dataflow import RNGDataFlow
from tensorpack.utils.argtools import shape2d
from tensorpack.utils.utils import get_rng
import os
import cv2
import numpy as np
import shutil
import sklearn.metrics
import glob2
import skimage.io
# import tensorflow as tf
from natsort import natsorted

class CustomDataSet(RNGDataFlow):
    """ Produce images read from a list of files as (h, w, c) arrays. """
    def __init__(self, folder, size=None, train_or_valid='train', channel=1, resize=None, debug=False, shuffle=False, hparams=None):
        super(CustomDataSet, self).__init__()
        self.folder = folder
        self.is_train = True if train_or_valid=='train' else False
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        if resize is not None:
            resize = shape2d(resize)
        self.resize = resize
        self.shuffle = shuffle
        self.hparams = hparams
        
        self.images = []
        self.labels = []

        if self.is_train:
            self.imageDir = os.path.join(folder, 'train', 'images')
            self.labelDir = os.path.join(folder, 'train', 'labels')
        else:
            self.imageDir = os.path.join(folder, 'test', 'images')
            self.labelDir = os.path.join(folder, 'test', 'labels')

        self.imageFiles = natsorted (glob2.glob(self.imageDir + '/*.*'))
        self.labelFiles = natsorted (glob2.glob(self.labelDir + '/*.*'))
        # print(self.imageFiles, self.labelFiles)
        self._size = min(size, len(self.imageFiles))
        print(self._size)
  
    def reset_state(self):
        self.rng = get_rng(self)   

    def __len__(self):
        return self._size

    def __iter__(self):
        # TODO
        indices = list(range(self.__len__()))
        if self.is_train:
            self.rng.shuffle(indices)

        for idx in indices:

            # image = self.images[idx].copy()
            # label = self.labels[idx].copy()
            image = cv2.imread (self.imageFiles[idx], cv2.IMREAD_GRAYSCALE)
            label = cv2.imread (self.labelFiles[idx], cv2.IMREAD_GRAYSCALE)

            label[label < 128] = 0
            label[label >= 128] = 255

            yield [image, label]

def DiceScore(output, target, smooth=1.0, epsilon=1e-7, axis=(2, 3)):
    """
    Compute mean dice coefficient over all abnormality classes.

    Args:
        output (Numpy tensor): tensor of ground truth values for all classes.
                                    shape: (batch, num_classes, x_dim, y_dim)
        target (Numpy tensor): tensor of predictions for all classes.
                                    shape: (batch, num_classes, x_dim, y_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator of dice coefficient.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant add to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_coefficient (float): computed value of dice coefficient.     
    """
    y_true = target
    y_pred = output
    dice_numerator = 2*np.sum(y_true*y_pred, axis=axis) + epsilon
    dice_denominator = (np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) + epsilon)
    dice_coefficient = np.mean(dice_numerator / dice_denominator)

    return dice_coefficient

class SoftDiceLoss(nn.Module):
    def init(self):
        super(SoftDiceLoss, self).init()

    def forward(self, output, target, smooth=1.0, epsilon=1e-7, axis=(0, 2, 3)):
        """
        Compute mean soft dice loss over all abnormality classes.

        Args:
            y_true (Torch tensor): tensor of ground truth values for all classes.
                                        shape: (batch, num_classes, x_dim, y_dim)
            y_pred (Torch tensor): tensor of soft predictions for all classes.
                                        shape: (batch, num_classes, x_dim, y_dim)
            axis (tuple): spatial axes to sum over when computing numerator and
                          denominator in formula for dice loss.
                          Hint: pass this as the 'axis' argument to the K.sum
                                and K.mean functions.
            epsilon (float): small constant added to numerator and denominator to
                            avoid divide by 0 errors.
        Returns:
            dice_loss (float): computed value of dice loss.  
        """
        y_true = target
        y_pred = output
        dice_numerator = 2*torch.sum(y_true*y_pred, dim=axis) + epsilon
        dice_denominator = (torch.sum(y_true*y_true, dim=axis) + torch.sum(y_pred*y_pred, dim=axis) + epsilon)
        dice_coefficient = torch.mean(dice_numerator / dice_denominator)
        
        dice_loss = 1 - dice_coefficient
        return dice_loss

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
        # if self.hparams.arch.lower() == 'unet' and self.hparams.backbone.lower() == 'vgg':
        #     self.model = UNet(input_channels=1, num_classes=self.hparams.types)
        # elif self.hparams.arch.lower() == 'uppnet' and self.hparams.backbone.lower() == 'vgg':
        #     self.model = UPPNet(input_channels=1, num_classes=self.hparams.types)
        # elif self.hparams.arch.lower() == 'fusionnet' and self.hparams.backbone.lower() == 'vgg':
        #     self.model = FusionNet(input_channels=1, num_classes=self.hparams.types)
        # else:
        #     ValueError
        # print(self.model.parameters())
        self.model = getattr(smp, self.hparams.arch)(self.hparams.backbone, 
                                                     classes=self.hparams.types, 
                                                     activation='sigmoid')
        print(self.model)
        self.model.encoder.layer0.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # logits -> nn.BCEWithLogitsLoss
        # logits -> sigmoid -> nn.BCELoss
        self.criterion = SoftDiceLoss()

    def forward(self, x):
        """Summary

        Args:
            x (TYPE): Description

        Returns:
            TYPE: Description
        """
        # y = (torch.tanh(self.model(x / 128.0 - 1.0)) / 2.0 + 0.5)*255.0
        y = self.model(x / 255.0) * 255.0
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
        images, target = batch
        output = self.forward(images)
        loss = self.criterion(output / 255.0, target / 255.0)

        tqdm_dict = {'train_loss': loss}
        result = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        viz = images
        for p in range(self.hparams.types):
            # print(viz.shape, target.shape)
            viz = torch.cat([viz, target, output], axis=-1)

        if batch_idx in range(5):
            self.logger.experiment.add_image(f'{prefix}_viz',
                                             torchvision.utils.make_grid(viz / 255.0, nrow=1, pad_value=1),
                                             dataformats='CHW', 
                                             global_step=self.global_step)
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
        images, target = batch
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
            viz = torch.cat([viz, target, output], axis=-1)

        if batch_idx in range(5):
            self.logger.experiment.add_image(f'{prefix}_viz',
                                             torchvision.utils.make_grid(viz / 255.0, nrow=1, pad_value=1),
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
        
        np_output = torch.cat([x[f'{prefix}_output'].squeeze_(0) for x in outputs], axis=0).to('cpu').numpy()
        np_target = torch.cat([x[f'{prefix}_target'].squeeze_(0) for x in outputs], axis=0).to('cpu').numpy()

        # Casting to binary
        result = {}
        tqdm_dict = {}
        tb_log = {}
        
        # Calculate 
        # macro dice-score: average dice-score of each image
        # micro dice-score: flatten all images and calculate

        if np_output.shape[0] > 0 and np_target.shape[0] > 0:
            macro_dice_score = DiceScore(np_output / 255.0, np_target / 255.0, axis=(2,3))
            micro_dice_score = DiceScore(np_output / 255.0, np_target / 255.0, axis=(0,2,3))
            tqdm_dict[f'{prefix}_macro_dice_score'] = f'{macro_dice_score:0.4f}'  
            tqdm_dict[f'{prefix}_micro_dice_score'] = f'{micro_dice_score:0.4f}'  
            tb_log[f'{prefix}_macro_dice_score'] = macro_dice_score 
            tb_log[f'{prefix}_micro_dice_score'] = micro_dice_score 
     
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
            size=np.inf,
            hparams=self.hparams
            )
        ds_train.reset_state()
        ag_train = [
            imgaug.Affine(shear=10, 
                interp=cv2.INTER_NEAREST),
            imgaug.Affine(translate_frac=(0.01, 0.02), 
                interp=cv2.INTER_NEAREST),
            imgaug.Affine(scale=(0.25, 1.0), 
                interp=cv2.INTER_NEAREST),
            imgaug.RotationAndCropValid(max_deg=10, interp=cv2.INTER_NEAREST),
            imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.8, 1.0), 
                    aspect_ratio_range=(0.8, 1.2),
                    interp=cv2.INTER_NEAREST, 
                    target_shape=self.hparams.shape),
            imgaug.Resize(self.hparams.shape, interp=cv2.INTER_NEAREST),
            imgaug.Flip(horiz=True, vert=False, prob=0.5),
            imgaug.Flip(horiz=False, vert=True, prob=0.5),
            imgaug.Transpose(prob=0.5),
            imgaug.Albumentations(AB.RandomRotate90(p=1)),
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
                                                    imgaug.Albumentations(AB.CLAHE(tile_grid_size=(32, 32), p=0.5)),
                                                        
                                                   ], 0)
        ds_train = AugmentImageComponents(ds_train, ag_train, [0, 1])
        
        ds_train = BatchData(ds_train, self.hparams.batch, remainder=True)
        if self.hparams.debug:
            ds_train = FixedSizeData(ds_train, 2)
        ds_train = MultiProcessRunner(ds_train, num_proc=4, num_prefetch=16)
        ds_train = PrintData(ds_train)

        ds_train = MapData(ds_train, lambda dp: [torch.tensor(dp[0][:,np.newaxis,:,:]).float(), 
                                                 torch.tensor(dp[1][:,np.newaxis,:,:]).float(),
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
            size=np.inf,
            hparams=self.hparams
            )

        ds_valid.reset_state()
        ag_valid = [
            imgaug.Resize(self.hparams.shape, interp=cv2.INTER_NEAREST),
            imgaug.ToFloat32(),
        ]
        ds_valid = AugmentImageComponent(ds_valid, [imgaug.Albumentations(AB.CLAHE(tile_grid_size=(32, 32), always_apply=True, p=1),),], 0)
        ds_valid = AugmentImageComponents(ds_valid, ag_valid, [0, 1])
        ds_valid = BatchData(ds_valid, self.hparams.batch, remainder=True)
        ds_valid = MultiProcessRunner(ds_valid, num_proc=4, num_prefetch=16)
        ds_valid = PrintData(ds_valid)
        ds_valid = MapData(ds_valid, lambda dp: [torch.tensor(dp[0][:,np.newaxis,:,:]).float(), 
                                                 torch.tensor(dp[1][:,np.newaxis,:,:]).float(),
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
            size=np.inf,
            hparams=self.hparams
            )

        ds_test.reset_state()
        ag_test = [
            imgaug.Resize(self.hparams.shape, interp=cv2.INTER_NEAREST),
            imgaug.ToFloat32(),
        ]
        # ds_test = AugmentImageComponent(ds_test, [imgaug.Albumentations(AB.CLAHE(tile_grid_size=(32, 32), always_apply=True, p=1)),], 0)
        ds_test = AugmentImageComponents(ds_test, ag_test, [0, 1])
        ds_test = BatchData(ds_test, self.hparams.batch, remainder=True)
        ds_test = MultiProcessRunner(ds_test, num_proc=4, num_prefetch=16)
        ds_test = PrintData(ds_test)
        ds_test = MapData(ds_test, lambda dp: [torch.tensor(dp[0][:,np.newaxis,:,:]).float(), 
                                                 torch.tensor(dp[1][:,np.newaxis,:,:]).float(),
                                                 ])
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
        parser.add_argument('-bb', '--backbone', metavar='backbone', default='vgg', 
                            help='backbone')
        parser.add_argument('--epochs', default=500, type=int, metavar='N',
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
    parent_parser.add_argument('--data', type=str, default=".", help='path to dataset')
    parent_parser.add_argument('--save', type=str, default="train_log", help='path to save output')
    parent_parser.add_argument('--info', type=str, default="train_log", help='path to logging output')
    parent_parser.add_argument('--gpus', type=int, default=1, help='how many gpus')
    parent_parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'), help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('--use-16bit', dest='use_16bit', action='store_true', help='if true uses 16 bit precision')
    parent_parser.add_argument('--percent_check', type=float, default=1.0, help="float/int. If float, % of tng epoch. If int, check every n batch")
    parent_parser.add_argument('--val_check_interval', type=float, default=1.0, help="float/int. If float, % of tng epoch. If int, check every n batch")
    parent_parser.add_argument('--fast_dev_run', action='store_true', default=False, help='fast_dev_run: runs 1 batch of train, test, val (ie: a unit test)')

    parent_parser.add_argument('--types', type=int, default=1)
    parent_parser.add_argument('--threshold', type=float, default=0.5)
    parent_parser.add_argument('--pathology', type=str, default='All')
    parent_parser.add_argument('--shape', type=int, default=512)
    # parent_parser.add_argument('--folds', type=int, default=5)

    # Inference purpose
    # parent_parser.add_argument('--load', help='load model')
    parent_parser.add_argument('--load', type=str, help='path to logging output')
    parent_parser.add_argument('--pred', action='store_true', help='run predict')
    parent_parser.add_argument('--eval', action='store_true', help='run offline evaluation instead of training')

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
                              str(hparams.backbone),
                              str(hparams.pathology),
                              str(hparams.shape),
                              str(hparams.types),
                              # str(hparams.folds),
                              # str(hparams.valid_fold_index),
                              'ckpt'),
        save_top_k=10,
        verbose=True,
        monitor='val_micro_dice_score',  # TODO
        mode='max'
    )

    trainer = pl.Trainer(
        train_percent_check=hparams.percent_check,
        val_percent_check=hparams.percent_check,
        test_percent_check=hparams.percent_check,
        num_sanity_val_steps=0,
        default_save_path=os.path.join(str(hparams.save),
                                       str(hparams.arch),
                                       str(hparams.backbone),
                                       str(hparams.pathology),
                                       str(hparams.shape),
                                       str(hparams.types),
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
        assert hparams.load
        model = ImageNetLightningModel(hparams).load_from_checkpoint(hparams.load)
        model.eval()
        trainer.test(model)
    elif hparams.pred:
        assert hparams.load
        model = ImageNetLightningModel(hparams).load_from_checkpoint(hparams.load)
        model.eval()

        imageFiles = natsorted (glob2.glob(hparams.data + '/*.*'))
        folder = 'result'
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
        for idx, imageFile in enumerate(imageFiles):
            image = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
            shape = image.shape
            image = cv2.resize(image, (hparams.shape, hparams.shape), cv2.INTER_AREA)
            image = image[np.newaxis, np.newaxis,:,:].astype(np.float32)
            image = torch.Tensor(torch.tensor(image))
            estim = model(image).detach().to('cpu').numpy()
            estim = np.squeeze(estim).astype(np.uint8)
            estim = cv2.resize(estim, shape[::-1], cv2.INTER_AREA)
            print(os.path.basename(imageFile), idx+1, len(imageFiles))
            cv2.imwrite(os.path.join(folder, os.path.basename(imageFile)), estim)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main(get_args())
