import os
import torch
import numpy as np
from config import cfg


def prune_model(model):
    compactor_idx = []
    compactor_channel = []
    model_state_dict = model.state_dict()
    compactor_thresh = cfg.MODEL.COMPACTOR_THRESH
    #print(model_state_dict.keys())

    for name in list(model_state_dict.keys()):
        if "Teacher" in name or "mask" in name:
            model_state_dict.pop(name)

    for name in list(model_state_dict.keys()):
        if 'stage0' in name:
            continue

        if 'compactor.weight' in name:
            param_data = model_state_dict[name]
            prune_data = param_data.detach().cpu().numpy()
            l2_prune_data = np.sqrt(np.sum(prune_data ** 2, axis=(1, 2, 3)))
            out_filter_thresh = np.where(l2_prune_data < compactor_thresh)[0]
            length = len(prune_data) - len(out_filter_thresh)
            if length == 0:
                l2_max = l2_prune_data.max()
                out_filter_thresh = np.where(l2_prune_data < l2_max)[0]
                compactor_channel.append(len(prune_data) - len(out_filter_thresh))
                compactor_idx.append(out_filter_thresh)

            elif length > 0:
                compactor_channel.append(length)
                compactor_idx.append(out_filter_thresh)
            prune_data = np.delete(prune_data, out_filter_thresh, axis=0)
            prune_data = torch.from_numpy(prune_data)
            model_state_dict[name] = prune_data


    compactor_count = 0
    print(compactor_channel)
    for name in list(model_state_dict.keys()):
        if 'num_batches_tracked' in name:
            model_state_dict.pop(name)
        elif 'stage0' in name:
            continue

        elif 'conv3.conv.weight' in name:
            param_data = model_state_dict[name]
            prune_data = param_data.detach().cpu().numpy()
            in_filter_thresh = compactor_idx[compactor_count]
           
            prune_data = np.delete(prune_data, in_filter_thresh, axis=1)
            prune_data = torch.from_numpy(prune_data)
            model_state_dict[name] = prune_data
            compactor_count += 1

    prune_channel = []
    for i in range(len(compactor_channel)):
        prune_channel.append(compactor_channel[i])
    print(prune_channel)
    torch.save(model_state_dict, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_prune_{}.pth'.format(cfg.TEST.WEIGHT)))
    with open('./log/prune_channel.txt', 'w') as f:
        f.write(str(prune_channel))
        f.close()