from pathlib import Path

import comet_ml
import torch
from torch.utils.data import DataLoader, Subset
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
parser.add_argument("--save-path", default='.')

args = parser.parse_args()

save_path=args.save_path

n_epochs = 10000
n_adv_per_enc = 1 #critic index
burnin_steps=2000 #n_epochs for the adversary
LR=1e-4
adv_LR=1e-4
batch_size=128
save_freq=1
n_sh_coeff = 28
dim_z = 32

scan_type_map = {
    "1200" : 0,
    "7T" : 1
}

dataset = TractoinfernoDataset(Path('/home/carl/data/tractoinferno/masked_full'), 'trainset', n_sh_coeff)
n_train = int(len(dataset) * 0.8)
train_dataset = Subset(dataset, torch.arange(n_train))
valid_dataset = Subset(dataset, torch.arange(start=n_train, end=len(dataset)))

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
dec_obj = arch.decoder( dim_z, vec_size, 1 )
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
global_step_valid = 0

loss_names = ['recon', 'kl', 'proj', 'marg', 'avg_g', 'adv_d']
train_metrics = {f'loss_{name}': torchmetrics.MeanMetric() for name in loss_names}
valid_metrics = {f'loss_{name}': torchmetrics.MeanMetric() for name in loss_names}

for epoch in range(n_epochs):

    # Training epoch
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

            train_metrics['loss_adv_d'].update(loss)
        else:

            optimizer.zero_grad()

            loss, (recon_loss,kl_loss, proj_loss, marg_loss, gen_adv_loss) = losses.enc_dec_training_step(
                enc_obj, dec_obj, adv_obj,
                x, c, center_vox_func,  x_subj_space, sh_mat, sh_weights,
                loss_weights, dim_z
            )

            loss.backward(retain_graph=True)
            optimizer.step()

            for name, l in zip(loss_names[:5], [recon_loss, kl_loss, proj_loss, marg_loss, gen_adv_loss]):
                train_metrics[name].update(l)

        comet_experiment.log_metric('train_loss', loss.item(), step=global_step)
        comet_experiment.log_metrics(
            {name: m.compute() for name, m in train_metrics.items()},
            step=global_step,
            prefix='train'
        )

        global_step += 1

    # Epoch end
    comet_experiment.log_metrics(
        {name: m.compute() for name, m in train_metrics.items()},
        epoch=epoch,
        prefix='ep_train'
    )
    for m in train_metrics:
        m.reset()

    # Validation epoch
    with torch.no_grad():
        for d_idx, batch in enumerate(tqdm(valid_loader)):
            x, c = batch
            c = c.unsqueeze(1)
            x = x.to(device).type(torch.float32)
            c = c.to(device)
            loss, (recon_loss, kl_loss, proj_loss, marg_loss, gen_adv_loss) = losses.enc_dec_training_step(
                enc_obj, dec_obj, adv_obj,
                x, c, center_vox_func, None, None, None,
                loss_weights, dim_z
            )
            for name, l in zip(loss_names[:-1], [recon_loss, kl_loss, proj_loss, marg_loss, gen_adv_loss]):
                valid_metrics[name].update(l)

    # Valid epoch end
    comet_experiment.log_metrics(
        {name: m.compute() for name, m in valid_metrics.items()},
        epoch=epoch,
        prefix='ep_valid'
    )
    for m in valid_metrics:
        m.reset()

    if save_path is not None and epoch % save_freq == 0:
        torch.save(
            {
                "enc":enc_obj.state_dict(),
                "dec":dec_obj.state_dict(),
                "adv":adv_obj.state_dict()
            },
            f"{save_path}/ckpt_{epoch}.pth"
        )
