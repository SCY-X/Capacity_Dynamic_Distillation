from yacs.config import CfgNode as CN
import os
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
# Model's Mode
_C.MODEL.MODE = 'train'
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Teacher Name of backbone
_C.MODEL.TEACHER_NAME = 'erhc_t_resnet101'
# Teacher model path
_C.MODEL.TEACHER_PATH = '/media/ub/62e066d5-f03e-404d-a93a-8cb471e6f2b3/xieyi/Teacher/MSMT_DRKD_T101.pth'
# Student Name of backbone
_C.MODEL.NAME = 'erhc_s_resnet101'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = '/media/ub/62e066d5-f03e-404d-a93a-8cb471e6f2b3/xieyi/Pretrained_Model/resnet101.pth'
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'finetune',
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# Frozen layers of backbone
_C.MODEL.FROZEN = -1
# Frozen layers of backbone
_C.MODEL.POOLING_METHOD = 'avg'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.USE_KD = True
_C.MODEL.KL_T = 2
_C.MODEL.KL_WEIGHT = 1.0

_C.MODEL.COMPACTOR_LASSO_WEIGHT = 0.004
_C.MODEL.COMPACTOR_THRESH = 0.00001

_C.MODEL.FIT_WEIGHT = 0.5

_C.MODEL.KD_METHOD = 'kl'
_C.MODEL.KD_WEIGHT = 0.0

_C.MODEL.MASK_RATIO = 0.5
_C.MODEL.MASK_DISTANCE = "cosine"
_C.MODEL.MASK_EPOCH = 20
_C.MODEL.QUEUE_SIZE = 24000
_C.MODEL.MASK_K = 2


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [320, 160]
# Size of the image during test
_C.INPUT.SIZE_TEST = [320, 160]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('msmt')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('/media/ub/62e066d5-f03e-404d-a93a-8cb471e6f2b3/xieyi/data')

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax_triplet'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 6

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = 'SGD'
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 120
# Base learning rate
_C.SOLVER.BASE_LR = 0.01
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = True
#the time learning rate of fc layer
_C.SOLVER.FC_LR_TIMES = 1.5
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss sampler method, option: batch_hard, batch_soft
_C.SOLVER.HARD_EXAMPLE_MINING_METHOD = 'batch_hard'
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.35
# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

#lr_scheduler
#lr_scheduler method, option WarmupMultiStepLR, WarmupCosineAnnealingLR
_C.SOLVER.LR_NAME = 'WarmupCosineAnnealingLR'
# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = [40, 70]

#Cosine annealing learning rate options
_C.SOLVER.DELAY_ITERS = 40
_C.SOLVER.ETA_MIN_LR = 1e-7

# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.1
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 10
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"
# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = _C.SOLVER.MAX_EPOCHS
# iteration of display training 0.003_0.7_10
_C.SOLVER.LOG_PERIOD = 270
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = _C.SOLVER.MAX_EPOCHS
_C.SOLVER.MIXED_PRECISION = True
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 96
_C.SOLVER.SEED = 2021
# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 512
# Whether using fliped feature for testing, option: 'on', 'off'
_C.TEST.FLIP_FEATS = 'off'
# Path to trained model
_C.TEST.WEIGHT = _C.SOLVER.MAX_EPOCHS
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.TEST_METRIC = 'cosine'
_C.TEST.RE_RANKING = False
# K1, K2, LAMBDA
_C.TEST.RE_RANKING_PARAMETER = [60, 10, 0.3]
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved 0.003_0.7_10 of trained model
_C.OUTPUT_DIR = "./log"
if not os.path.isdir(_C.OUTPUT_DIR):
    os.makedirs(_C.OUTPUT_DIR)
