# /usr/bin/env python
# from enum import Enum
from easydict import EasyDict as edict
from enum import Enum

import copy


class phase(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


""" Configuration file."""

######### ProSR #########
prosr_params = \
    edict({
        'train': {
            'dataset': {
                'path': {
                    'source':'',
                    'target':'data/datasets/DIV2K/DIV2K_train_HR'
                    # 'target':'data/datasets/Ensemble/**'
                },
                'downscale':False,
                'mean': [0.4488, 0.4371, 0.4040],  # mean value to extract from the (0, 1) image values
                'stddev': [0.0039215, 0.0039215, 0.0039215]  # multiply the image value by this factor, resulting value range of image [-127.5, 127.5]
            },
            'epochs': 500,  #
            'batch_size': 16,
            'growing_steps': [0.12, 0.25, 0.45, 0.6, 1.00],
            'lr_schedule_patience': 30,
            'lr': 0.0001,
            'lr_decay': 0.5,
            'smallest_lr': 1e-5,
            'l1_loss_weight': 1.0,
            ############# output settings ##############
            'io': {
                'save_model_freq':20,
                'eval_epoch_freq': 10,
                'print_errors_freq': 100
            },
        },
        'G': {
            'class_name': 'ProSR',
            'residual_denseblock': True,  # ProSR_l and ProSRGan uses residual links, ProSR_l doesn't
            'max_scale': 8,  # cooresponds to max(prosr_params.data.scale)
            # densenet hyperparameters
            'num_init_features': 160,
            'growth_rate': 40,
            # architecture for each pyramid level
            # len(level_config) means the number of pyramids (equals the number of scales)
            # len(level_config(i)) means the number of DCUs in pyramid i
            # level_config(i)(j) means the number of dense layers in j-th DCU of the i-th pyramid.
            'level_config': [[8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8], [8]],
            # maximum number of features before subpixel upsampling
            'max_num_feature': 312,
            'ps_woReLU': False,  # ps without relu
            'level_compression': -1,  # used between pyramid levels, if <0: compress to same as input
            'bn_size': 4,
            'res_factor': 0.2,  # scale residual
        },
        'test': {
            'fast_validation':-1, #-1 full validation, 0 no validation, x: max files validation
            'dataset': {
                'path': {
                    'source':'',
                    'target':'data/datasets/Set14/HR'

                },
                'downscale':False,
                'mean': [0.4488, 0.4371, 0.4040],  # mean value to extract from the (0, 1) image values
                'stddev': [0.0039215, 0.0039215, 0.0039215]  # multiply the image value by this factor, resulting value range of image [-127.5, 127.5]
            },
        },
        'data': {
            'scale': [2, 4, 8],
            'input_size': [48, 36, 24]  # reduce input size for 4x and 8x to save memory
        }
    })
prosrs_params = copy.deepcopy(prosr_params)
prosrs_params.G.level_config = [[6, 6, 6, 6], [6, 6], [6]]
prosrs_params.G.num_init_features = 24
prosrs_params.G.growth_rate = 12
prosrs_params.G.block_compression = 0.4
prosrs_params.G.level_compression = 0.5
prosrs_params.G.residual_denseblock = False
prosrs_params.G.res_factor = 1.0

prosrgan_params = copy.deepcopy(prosr_params)
prosrgan_params.D = edict({
    'input_residual': True,  # discriminates residual
    'scale_overhead': True,  # pyramid discriminator
    'warmup_epochs': 2, # train discriminator 2 epochs before iterating D and G
    'update_freq': 2, # every D update corresponds to twice G updates
    'use_lsgan': True,  # use least square GAN
    'ndf': 64,  # discriminator feature number
})
prosrgan_params.train.D_lr = 0.0001
prosrgan_params.train.vgg_loss_weight = [0.5, 2]
prosrgan_params.train.gan_loss_weight = 1
prosrgan_params.train.l1_loss_weight = 0
prosrgan_params.train.vgg = [2, 4]
prosrgan_params.train.vgg_mean_pool = True

debug_params = copy.deepcopy(prosrs_params)
debug_params.train.io.eval_epoch_freq = 1
debug_params.train.io.print_errors_freq = 10
debug_params.train.io.save_model_freq = 5
debug_params.train.dataset.path.target = 'data/datasets/DIV2K/DIV2K_debug_HR'
debug_params.train.epochs = 10
debug_params.test.fast_validation = 2


edsr_params = copy.deepcopy(prosr_params)
edsr_params.data.scale = [8]
edsr_params.data.input_size = [24]
edsr_params.G.class_name = 'EDSR'
edsr_params.G.num_blocks = 36
edsr_params.G.kernel_size= 3
edsr_params.G.upscale_factor = edsr_params.data.scale[0]

# gandebug_params = copy.deepcopy(prosrgan_params)
# gandebug_params.train.io.eval_epoch_freq = 1
# gandebug_params.train.io.print_errors_freq = 10
# gandebug_params.train.io.save_model_freq = 5
# gandebug_params.train.dataset.path.target = 'data/datasets/DIV2K/DIV2K_debug_HR'
# gandebug_params.train.epochs = 10
# gandebug_params.test.fast_validation = 2
