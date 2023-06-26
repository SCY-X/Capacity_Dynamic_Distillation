import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
from ptflops import get_model_complexity_info
import torch

def test(val_loader, num_query, num_classes, deploy_flag):
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()


    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_model_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(cfg.TEST.WEIGHT))
    model = make_model(cfg, num_class=num_classes, deploy_flag=deploy_flag, prune=None)
    if torch.cuda.device_count() > 0:
        model.cuda()


    model.load_param_test(train_model_path)


    # from processor.prune_model import prune_model
    # prune_model(model)
    # exit()

    do_inference(cfg, model, val_loader, num_query)
    model.eval()
    image_size = (3, cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1])
    macs, _ = get_model_complexity_info(model, image_size, as_strings=True,
                                                print_per_layer_stat=False, verbose=True)
    params = (sum([param.nelement() for param in model.parameters()]) / 1e6) - 0.53
    
    
    logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    logger.info('{:<30}  {:.2f}'.format('Number of parameters: ', params))
    with open('%s/test_acc.txt' % cfg.OUTPUT_DIR, 'a') as test_file:
        test_file.write('{:<30}  {:<8}\n'.format('Computational complexity: ', macs))
        test_file.write('{:<30}  {:.2f}\n'.format('Number of parameters: ', params))
        test_file.write("--------------\n")

    #os.remove(train_model_path)

def test_prune(val_loader, num_query, num_classes, deploy_flag, test_rep_prune=False):
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()


    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    f = open('./log/prune_channel.txt', 'r')
    prune_channel = f.readlines()
    prune_channel = prune_channel[0].strip('[').strip(']').split(',')
    prune_channel = [int(channel) for channel in prune_channel]
    print(prune_channel)

    model = make_model(cfg, num_class=num_classes, deploy_flag=deploy_flag, prune=prune_channel)
    if test_rep_prune:
        train_model_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_rep_prune_{}.pth'.format(cfg.TEST.WEIGHT))
        model.load_param_test(train_model_path)
        if torch.cuda.device_count() > 0:
            model.cuda()
        do_inference(cfg, model, val_loader, num_query)

        model.eval()
        image_size = (3, cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1])
        macs, _ = get_model_complexity_info(model, image_size, as_strings=True,
                                                print_per_layer_stat=False, verbose=True)
        params = (sum([param.nelement() for param in model.parameters()]) / 1e6) - 0.53
        
        logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        logger.info('{:<30}  {:.2f}'.format('Number of parameters: ', params))
        with open('%s/test_acc.txt' % cfg.OUTPUT_DIR, 'a') as test_file:
            test_file.write('{:<30}  {:<8}\n'.format('Computational complexity: ', macs))
            test_file.write('{:<30}  {:.2f}\n'.format('Number of parameters: ', params))
            test_file.write("--------------\n")


    else:
        train_model_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_prune_{}.pth'.format(cfg.TEST.WEIGHT))
        model.load_param_test(train_model_path)
        if torch.cuda.device_count() > 0:
            model.cuda()
        do_inference(cfg, model, val_loader, num_query)
        for m in model.modules():
            m.requires_grad_(False)
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_rep_prune_{}.pth'.format(cfg.TEST.WEIGHT)))
        os.remove(train_model_path)