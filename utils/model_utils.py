import copy
import os
import torch
from collections import OrderedDict
import copy

def save_model(args, model, optimizer, current_epoch):
    out = os.path.join(args.model_path,
                       f"{args.model_type}_{str(args.backbone).lower()}_checkpoint_{current_epoch}.tar")
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)


def load_model(net, checkpoint, filter_team_classifier=False):
    # state_dict = OrderedDict()
    state_dict = copy.deepcopy(net.state_dict())
    for k, v in checkpoint['net'].items():
        if filter_team_classifier and 'team_classifier' in k:
            continue
        if 'module' in k:
            name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
        else:
            name = k
        state_dict[name] = v
    # model load
    net.load_state_dict(state_dict)
    return net
