import torch
import torch_tensorrt
import numpy as np
import os
from modules import resnet, shufflenetv2, efficientnet
from modules.models import PRTreIDTeamClassifier
from utils.model_utils import load_model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model_path = 'weights/resnet18_aug:bright05:_NMI:05074_ARI:06174_F:08099_ACC:08931_withoutrole.tar'
model_path = 'weights/resnet18_aug:bright05:_NMI:05074_ARI:06174_F:08099_ACC:08931.tar'
# model_path = 'save/Hockey/prtrei_resnet18_checkpoint_80_with_role.tar'

if 'resnet' in os.path.basename(model_path):
    backbone = resnet.get_resnet('ResNet18')
elif 'shufflenetv2' in os.path.basename(model_path):
    backbone = shufflenetv2.get_shufflenet('ShuffleNetV2')
elif 'efficientnet' in os.path.basename(model_path):
    backbone = efficientnet.get_efficientnet('EfficientNetB0')
else:
    raise NotImplementedError

model = PRTreIDTeamClassifier(backbone=backbone, num_teams=12, attention_enable=True)
model = model.to(device)

checkpoint = torch.load(model_path)
load_model(model, checkpoint, filter_team_classifier=True)
model.eval()

inputs = torch_tensorrt.Input(
    min_shape=(1, 3, 256, 128),
    opt_shape=(32, 3, 256, 128),
    max_shape=(64, 3, 256, 128))
trt_model = torch_tensorrt.compile(
    model,
    inputs=inputs,
    enabled_precisions={torch.float}
)
save_model_path = model_path.replace('.tar', '.ts')
x = torch.randn((1, 3, 256, 128)).to(device)

scripted_model = torch.jit.trace(trt_model, x)
scripted_model.save(save_model_path)

loaded_trt_model = torch.jit.load(save_model_path).to(device)
f_trt, _, mask_trt, role_cls_trt = loaded_trt_model(x)
f, _, mask, role_cls = model(x)

print(np.max(f_trt.detach().cpu().numpy() - f.detach().cpu().numpy()))