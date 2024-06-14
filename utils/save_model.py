import os
import torch


def save_model(args, model, optimizer, current_epoch):
    out = os.path.join(args.model_path,
                       f"{args.model_type}_{str(args.backbone).lower()}_checkpoint_{current_epoch}.tar")
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)
