import torch
YOUR_CKPT_PATH = None
file_path = '/data3/yangpx/Project_.5/VAD_stage2/unc0.025_traj_0.3_0.2_plan_1.0_0.5_planloss_1.5_1.5_1.0_fde_sinemb2/epoch_10.pth'
model = torch.load(file_path, map_location='cpu')
all = 0
for key in list(model['state_dict'].keys()):
    all += model['state_dict'][key].nelement()
print(all)
