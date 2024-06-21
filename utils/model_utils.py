import copy
import os
import torch
import torch.nn as nn


def save_model(args, model, optimizer, current_epoch):
    out = os.path.join(args.model_path,
                       f"{args.model_type}_{str(args.backbone).lower()}_checkpoint_{current_epoch}.tar")
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)


def load_model(net, checkpoint, filter_team_classifier=False):
    # state_dict = OrderedDict()
    state_dict = copy.deepcopy(net.state_dict())
    for k, v in checkpoint['net'].items():
        if filter_team_classifier and 'team_classifier.classifier.weight' in k:
            continue
        if 'module' in k:
            name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
        else:
            name = k
        state_dict[name] = v
    # model load
    net.load_state_dict(state_dict)
    return net


def freeze_model(net):
    # Model freeze
    for name, m in net.named_modules():
        if 'role_classifier' in name:
            continue
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)
            m.affine = False
            m.track_running_stats = False
        m.eval()
    for name, param in net.named_parameters():
        if 'role_classifier' in name:
            continue
        else:
            param.requires_grad = False


def torch_time_checker(func):
    def wrapper(*args, **kwargs):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        starter.record()
        # model inference
        result = func(*args, **kwargs)
        ender.record()

        # wait for gpu sync
        torch.cuda.synchronize()
        print(f'function time: {round(starter.elapsed_time(ender), 5)}ms')
        return result

    return wrapper
