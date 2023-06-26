import os
import torch
from config import cfg
from datasets.make_dataloader import make_dataloader
from tools.train import train
from tools.test import test, test_prune

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID


if __name__ == '__main__':

    train_loader, val_loader, num_query, num_classes= make_dataloader(cfg)
    if cfg.MODEL.MODE == 'train':
######################################### resume model ###################################
        train(train_loader, num_classes, deploy_flag=False)

        with torch.no_grad():
            test(val_loader, num_query, num_classes, deploy_flag=True)
            test_prune(val_loader, num_query, num_classes, deploy_flag=False)
            test_prune(val_loader, num_query, num_classes, deploy_flag=True, test_rep_prune=True)

    if cfg.MODEL.MODE == 'evaluate':
        with torch.no_grad():
            test(val_loader, num_query, num_classes, deploy_flag=True)
            test_prune(val_loader, num_query, num_classes, deploy_flag=False)
            test_prune(val_loader, num_query, num_classes, deploy_flag=True, test_rep_prune=True)




