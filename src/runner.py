from pathlib import Path

import comet_ml
import torch
#import loader
import arch
import losses
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from src.tractoinferno import TractoinfernoDataset

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

import argparse

parser = argparse.ArgumentParser(description=\
    "runs inv-rep auto-encoder training"
)

parser.add_argument("--hcp-zip-path")
parser.add_argument("--save-path",default=None)

args = parser.parse_args()

#PATH_TO_HCP_DATA=args.hcp_zip_path
save_path=args.save_path

n_epochs = 10000
n_adv_per_enc = 1 #critic index
burnin_steps=2000 #n_epochs for the adversary
LR=1e-4
adv_LR=1e-4
batch_size=128
save_freq=5
n_sh_coeff = 28
dim_z = 32

scan_type_map = {
    "1200" : 0,
    "7T" : 1
}

dataset = TractoinfernoDataset(Path('/home/carl/data/tractoinferno/masked_full'), 'trainset', n_sh_coeff)

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=4
)

#center_vox_func = train_iterator.get_center_voxel_function()
center_vox_func = None

#vec_size = 322
vec_size = n_sh_coeff * 7

enc_obj = arch.encoder( vec_size, dim_z )
dec_obj = arch.decoder( dim_z, vec_size, 1 )
adv_obj = arch.adv( vec_size, 1 )

enc_obj.to(device)
dec_obj.to(device)
adv_obj.to(device)

#should use itertools chain
optimizer = torch.optim.Adam(
    list(enc_obj.parameters()) + list(dec_obj.parameters()), lr=LR
)
adv_optimizer = torch.optim.Adam(adv_obj.parameters(), lr=adv_LR)

loss_weights = {
    "recon" : 1.0,
    "prior" : 1.0,
    "projection" : 1.0,
    "marg" : 0.01,
    "adv" : 10.0
}

comet_experiment = comet_ml.Experiment(project_name='harmon_moyer')
comet_experiment.log_parameters({
    'n_adv_per_enc': n_adv_per_enc,
    'burnin_steps': burnin_steps,
    'LR': LR,
    'adv_LR': adv_LR,
    'batch_size': batch_size,
    'save_freq': save_freq,
    'n_sh_coeff': n_sh_coeff,
    'dim_z': dim_z
})
comet_experiment.log_parameters(loss_weights, prefix='loss_weight')

print('Training starts')

global_step = 0
for epoch in range(n_epochs):
    recon_loss = torch.tensor(0)
    kl_loss = torch.tensor(0)
    proj_loss = torch.tensor(0)
    marg_loss = torch.tensor(0)
    gen_adv_loss = torch.tensor(0)

    for d_idx, batch in enumerate(tqdm(train_loader)):
        #print(f"batch {d_idx}", flush=True)

        x, c = batch
        c = c.unsqueeze(1)
        x_subj_space = sh_mat = sh_weights = None

        x = x.to(device).type(torch.float32)
        #x_subj_space = x_subj_space.to(device)
        #sh_mat = sh_mat.to(device)
        #sh_weights = sh_weights.to(device)
        c = c.to(device)

        if (epoch == 0 and d_idx < burnin_steps) or d_idx % (n_adv_per_enc+1) > 0:
            adv_optimizer.zero_grad()
            
            loss = losses.adv_training_step(
                enc_obj, dec_obj, adv_obj, x, c
            )

            loss.backward(retain_graph=True)
            adv_optimizer.step()
        else:

            optimizer.zero_grad()

            loss, (recon_loss,kl_loss, proj_loss, marg_loss, gen_adv_loss) = losses.enc_dec_training_step(
                enc_obj, dec_obj, adv_obj,
                x, c, center_vox_func,  x_subj_space, sh_mat, sh_weights,
                loss_weights, dim_z
            )

            loss.backward(retain_graph=True)
            optimizer.step()

        global_step += 1

        #del x, x_subj_space, sh_mat, sh_weights, c

        comet_experiment.log_metrics({
            'loss': loss.item(),
            'loss_recon': recon_loss.item(),
            'loss_kl': kl_loss.item(),
            'loss_marg': marg_loss.item(),
            'loss_adv': gen_adv_loss.item()
        }, step=global_step)

    if save_path is not None and epoch % save_freq == 0:
        torch.save(
            {
                "enc":enc_obj.state_dict(),
                "dec":dec_obj.state_dict(),
                "adv":adv_obj.state_dict()
            },
            f"{save_path}/{epoch}.pth"
        )
