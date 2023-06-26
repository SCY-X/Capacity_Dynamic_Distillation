import torch.nn as nn
import torch.nn.functional as F
import logging
import math
import config 
import torch
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .kl_loss import KL, Fitnet_loss
from .MKD import MKD
from .ACRD import ACRD
from config import cfg

class Make_Loss(nn.Module):
    def __init__(self, cfg, num_classes):
        super(Make_Loss, self).__init__()
        logger = logging.getLogger("reid_baseline.train")
        if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
            if cfg.MODEL.NO_MARGIN:
                self.triplet = TripletLoss(mining_method='batch_soft')
                logger.info("using soft margin triplet loss for training, mining_method: batch_soft")
            else:
                self.triplet = TripletLoss(cfg.SOLVER.MARGIN, mining_method=cfg.SOLVER.HARD_EXAMPLE_MINING_METHOD)
                logger.info("using Triplet Loss for training with margin:{}, mining_method:{}".format(cfg.SOLVER.MARGIN, cfg.SOLVER.HARD_EXAMPLE_MINING_METHOD))
        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            self.id_loss = CrossEntropyLabelSmooth(num_classes=num_classes)
            logger.info("label smooth on, numclasses:{}".format(num_classes))
        else:
            self.id_loss = F.cross_entropy

        self.KD_METHOD = cfg.MODEL.KD_METHOD
        self.KL_Loss = KL(cfg.MODEL.KL_T)
        self.DL = Fitnet_loss

        if self.KD_METHOD == 'kl':
            pass

        # elif self.KD_METHOD == 'acrd':
        #     self.KD_Loss = ACRD(cfg.MODEL.P, cfg.MODEL.TOPK, cfg.MODEL.ACRD_ALPHA, cfg.MODEL.ACRD_BETA,
        #                         cfg.MODEL.ACRD_MODE, cfg.MODEL.POOL)
        #     logger.info('using ACRD alpha weight is {}, ACRD beta weight is {}'.format(cfg.MODEL.ACRD_ALPHA,
        #                                                                                cfg.MODEL.ACRD_BETA))
        else:
            raise NotImplementedError(self.KD_METHOD)

        logger.info('using {} method as KD Loss and KD weight is {}'.format(self.KD_METHOD, cfg.MODEL.KD_WEIGHT))

        self.id_loss_weight = cfg.MODEL.ID_LOSS_WEIGHT
        self.tri_loss_weight = cfg.MODEL.TRIPLET_LOSS_WEIGHT
        self.kl_loss_weight = cfg.MODEL.KL_WEIGHT
        self.dl_loss_weight = cfg.MODEL.FIT_WEIGHT

        self.kd_loss_weight = cfg.MODEL.KD_WEIGHT
        self.lasso_weight = cfg.MODEL.COMPACTOR_LASSO_WEIGHT


    def forward(self, score, tri_feat, compactor_list, feat_map, target):
        ce_loss = self.id_loss_weight * self.id_loss(score[0], target)
        triplet_loss = self.tri_loss_weight * self.triplet(tri_feat, target)
        compactor_loss = self.lasso_weight * sum(compactor_list)
        kl_loss = self.kl_loss_weight * self.KL_Loss(score[0], score[1])

        dl_loss = self.dl_loss_weight * Fitnet_loss(feat_map)


        if self.KD_METHOD == 'kl':
            kd_loss = torch.Tensor([0.0]).cuda()

        # elif self.KD_METHOD == 'acrd':
        #     kd_loss = self.KD_Loss([s_feat[-1]], [t_feat[-1]], statistics)

        kd_loss = self.kd_loss_weight * kd_loss
        total_loss = ce_loss + triplet_loss + kl_loss + dl_loss + kd_loss
        return total_loss, ce_loss, triplet_loss, kl_loss, compactor_loss, dl_loss, kd_loss
