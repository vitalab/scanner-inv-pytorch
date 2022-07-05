from pathlib import Path

import comet_ml
import torch
from torch.utils.data import DataLoader
import arch
import losses
import torchmetrics
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
parser.add_argument("--save-path", default='./checkpoints')
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()

save_path=args.save_path

n_epochs = 10000
n_adv_per_enc = 1 #critic index
burnin_epochs=1 #n_epochs for the adversary
LR=1e-4
adv_LR=1e-4
batch_size=128
save_freq=1
n_sh_coeff = 1 if args.debug else 28
dim_z = 32
num_sites = 6

scan_type_map = {
    "1200" : 0,
    "7T" : 1
}

# FIXME add command line arg for dataset path
train_dataset = TractoinfernoDataset(Path('/home/carl/data/tractoinferno/masked_full'), 'trainset', n_sh_coeff, debug=args.debug)
valid_dataset = TractoinfernoDataset(Path('/home/carl/data/tractoinferno/masked_full'), 'validset', n_sh_coeff, debug=args.debug)

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
    'burnin_epochs': burnin_epochs,
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

gen_step_loss_names = [f'loss_{name}' for name in ['recon', 'kl', 'marg', 'adv_g']]
train_metrics = {name: torchmetrics.MeanMetric() for name in gen_step_loss_names + ['loss_adv_d']}
valid_metrics = {name: torchmetrics.MeanMetric() for name in gen_step_loss_names + ['loss_adv_d']}

for epoch in range(n_epochs):

    # Training epoch
    for d_idx, batch in enumerate(tqdm(train_loader, desc=f'Train epoch {epoch}')):
        x, c = batch
        x_subj_space = sh_mat = sh_weights = None

        x = x.to(device).type(torch.float32)
        c = c.to(device)

        if epoch < burnin_epochs or d_idx % (n_adv_per_enc+1) > 0:
            adv_optimizer.zero_grad()
            
            loss = losses.adv_training_step(
                enc_obj, dec_obj, adv_obj, x, c
            )

            loss.backward(retain_graph=True)
            adv_optimizer.step()

            train_metrics['loss_adv_d'].update(loss.item())
            comet_experiment.log_metric('train_loss_adv_d', loss.item(), step=global_step)
        else:

            optimizer.zero_grad()

            loss, separate_losses = losses.enc_dec_training_step(
                enc_obj, dec_obj, adv_obj, x, c, loss_weights, dim_z
            )

            loss.backward(retain_graph=True)
            optimizer.step()

            for name, l in zip(gen_step_loss_names, separate_losses):
                train_metrics[name].update(l.item())
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
            loss, separate_losses = losses.enc_dec_training_step(
                enc_obj, dec_obj, adv_obj, x, c, loss_weights, dim_z
            )
            for name, l in zip(gen_step_loss_names, separate_losses):
                valid_metrics[name].update(l.item())

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
            f"{save_path}/ckpt_{epoch}.pth"
        )
