from pathlib import Path

import comet_ml
import torch
from torch.utils.data import DataLoader
import arch
import losses
import torchmetrics
from tqdm import tqdm

from src.tractoinferno import TractoinfernoDataset

import argparse

parser = argparse.ArgumentParser(description=\
    "runs inv-rep auto-encoder training"
)

parser.add_argument("data_path", type=Path)
parser.add_argument("--hcp-zip-path")
parser.add_argument("--save-path", default='./checkpoints')
parser.add_argument("--debug", action="store_true")
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--burnin_eps", type=int, default=1)
parser.add_argument("--lwreconfactor", type=float, default=1.0)
parser.add_argument("--use_adv", choices=['y', 'n'], default='y')

args = parser.parse_args()

if not args.cpu and torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

save_path=args.save_path

n_epochs = args.epochs
n_adv_per_enc = 1 #critic index
burnin_epochs=args.burnin_eps #n_epochs for the adversary
LR=1e-4
adv_LR=1e-4
batch_size=128
save_freq=10
n_sh_coeff = 1 if args.debug else 28
dim_z = 32
num_sites = 6

scan_type_map = {
    "1200" : 0,
    "7T" : 1
}

# FIXME add command line arg for dataset path
train_dataset = TractoinfernoDataset(args.data_path, 'trainset', n_sh_coeff, debug=args.debug)
valid_dataset = TractoinfernoDataset(args.data_path, 'validset', n_sh_coeff, debug=args.debug)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,  # Do not shuffle, the dataset has been shuffled. shuffle=True is much slower.
    pin_memory=True,
    num_workers=4
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=4
)

center_vox_func = None

#vec_size = 322
vec_size = n_sh_coeff * 7

enc_obj = arch.encoder( vec_size, dim_z )
dec_obj = arch.decoder( dim_z, vec_size, num_sites )
adv_obj = arch.adv( vec_size, 1 )

enc_obj.to(device)
dec_obj.to(device)
adv_obj.to(device)

optimizer = torch.optim.Adam(
    list(enc_obj.parameters()) + list(dec_obj.parameters()), lr=LR
)
adv_optimizer = torch.optim.Adam(adv_obj.parameters(), lr=adv_LR)

use_adv = args.use_adv == 'y'
loss_weights = {
    "recon" : 1.0,
    "prior" : 1.0 / args.lwreconfactor,
    "marg" : 0.01 / args.lwreconfactor,
    "adv_g" : 10.0 / args.lwreconfactor if use_adv else 0.0
}

comet_experiment = comet_ml.Experiment(project_name='harmon_moyer')
comet_experiment.log_parameters({
    'n_adv_per_enc': n_adv_per_enc,
    'burnin_epochs': burnin_epochs,
    'epochs': args.epochs,
    'lwreconfactor': args.lwreconfactor,
    'use_adv': use_adv,
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
global_step_valid = 0

gen_step_loss_names = [f'loss_{name}' for name in ['recon', 'prior', 'marg', 'adv_g']]
train_metrics = {name: torchmetrics.MeanMetric(nan_strategy='error') for name in gen_step_loss_names + ['loss_adv_d']}
valid_metrics = {name: torchmetrics.MeanMetric(nan_strategy='error') for name in gen_step_loss_names + ['loss_adv_d']}

for epoch in range(n_epochs):

    # Training epoch
    for d_idx, batch in enumerate(tqdm(train_loader, desc=f'Train epoch {epoch}')):
        x, c = batch
        x_subj_space = sh_mat = sh_weights = None

        x = x.to(device).type(torch.float32)
        c = c.to(device)

        if use_adv and (epoch < burnin_epochs or d_idx % (n_adv_per_enc+1) > 0):
            adv_optimizer.zero_grad()
            
            loss = losses.adv_training_step(
                enc_obj, dec_obj, adv_obj, x, c, num_sites=num_sites
            )

            loss.backward()
            adv_optimizer.step()

            train_metrics['loss_adv_d'].update(loss.item())
            comet_experiment.log_metric('train_loss_adv_d', loss.item(), step=global_step)
        else:

            def forward_pass(tmp_loss_weights):
                loss, separate_losses, z_mu, x_recon = losses.enc_dec_training_step(
                    enc_obj, dec_obj, adv_obj, x, c, tmp_loss_weights, dim_z, num_sites=num_sites
                )
                return loss, separate_losses, z_mu, x_recon

            # Inspect gradients for each loss term
            dmu = {}
            drecon = {}
            grad_w_encoder = {}
            grad_w_decoder = {}
            for loss_name in loss_weights.keys():
                # Zero out all loss weights, except one
                tmp_loss_weights = {name: value if name == loss_name else 0.0 for name, value in loss_weights.items()}
                # Run forward pass
                optimizer.zero_grad()
                loss, _, z_mu, x_recon = forward_pass(tmp_loss_weights)
                z_mu.retain_grad()
                x_recon.retain_grad()
                loss.backward()

                # === Store gradient norms
                # At output of encoder
                dmu[loss_name] = torch.linalg.norm(z_mu.grad)
                # At output of decoder
                drecon[loss_name] = torch.linalg.norm(x_recon.grad)
                # On weights of encoder
                dw_enc = torch.cat([p.grad.flatten() for p in enc_obj.parameters()])
                grad_w_encoder[loss_name] = torch.linalg.norm(dw_enc)
                # On weights of decoder
                dw_dec = torch.cat([p.grad.flatten() for p in dec_obj.parameters()])
                grad_w_decoder[loss_name] = torch.linalg.norm(dw_dec)
            # Log values of interest
            comet_experiment.log_metrics({
                'dlrecon_dmu': dmu['recon'],
                'dlprior_dmu': dmu['prior'],
                'dlmarg_dmu': dmu['marg'],
                'dladv_dmu': dmu['adv_g'],
                'dlrecon_dxrecon': drecon['recon'],
                'dladv_dxrecon': drecon['adv_g'],

                'dlrecon_dwenc': grad_w_encoder['recon'],
                'dlprior_dwenc': grad_w_encoder['prior'],
                'dlmarg_dwenc': grad_w_encoder['marg'],
                'dladv_dwenc': grad_w_encoder['adv_g'],
                'dlrecon_dwdec': grad_w_decoder['recon'],
                'dladv_dwdec': grad_w_decoder['adv_g'],
            })

            # Run real forward pass and update
            optimizer.zero_grad()
            loss, separate_losses, _, _ = forward_pass(loss_weights)
            loss.backward()
            optimizer.step()

            for name, l in zip(gen_step_loss_names, separate_losses):
                try:
                    train_metrics[name].update(l.item())
                except RuntimeError:
                    print('While updating metric', name)
                    raise
                comet_experiment.log_metric(f'train_{name}', l.item(), step=global_step)

        comet_experiment.log_metric('train_loss', loss.item(), step=global_step)
        global_step += 1

    # Epoch end
    comet_experiment.log_metrics(
        {name: m.compute() for name, m in train_metrics.items() if m._update_called},
        epoch=epoch,
        prefix='ep_train'
    )
    for m in train_metrics.values():
        m.reset()

    # Validation epoch
    with torch.no_grad():
        for d_idx, batch in enumerate(tqdm(valid_loader, desc=f'Valid epoch {epoch}')):
            x, c = batch
            x = x.to(device).type(torch.float32)
            c = c.to(device)
            loss, separate_losses, _, _ = losses.enc_dec_training_step(
                enc_obj, dec_obj, adv_obj, x, c, loss_weights, dim_z, num_sites=num_sites
            )
            for name, l in zip(gen_step_loss_names, separate_losses):
                try:
                    valid_metrics[name].update(l.item())
                except RuntimeError:
                    print('While updating metric', name)
                    raise

    # Valid epoch end
    comet_experiment.log_metrics(
        {name: m.compute() for name, m in valid_metrics.items() if m._update_called},
        epoch=epoch,
        prefix='ep_valid'
    )
    for m in valid_metrics.values():
        m.reset()

    if save_path is not None and epoch % save_freq == 0:
        Path(save_path).mkdir(exist_ok=True)
        torch.save(
            {
                "enc":enc_obj.state_dict(),
                "dec":dec_obj.state_dict(),
                "adv":adv_obj.state_dict()
            },
            f"{save_path}/ckpt__{comet_experiment.name}__{epoch}.pth"
        )
