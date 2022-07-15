from pathlib import Path

import comet_ml
import torch
from torch.utils.data import DataLoader
import arch
import losses
import torchmetrics
from tqdm import tqdm
import optuna

from src.tractoinferno import TractoinfernoDataset

import argparse

parser = argparse.ArgumentParser(description=\
    "runs inv-rep auto-encoder training"
)

parser.add_argument("--hcp-zip-path")
parser.add_argument("--save-path", default='./checkpoints')
parser.add_argument("--debug", action="store_true")
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--optuna_study", default=None)
parser.add_argument("--optuna_db", default='optuna_db.sqlite')
parser.add_argument("--optuna_trials", default=1)


args = parser.parse_args()

if not args.cpu and torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

save_path=args.save_path

default_hparams = {
    'n_epochs': 1,
    'n_adv_per_enc': 1,
    'burnin_epochs': 1,
    'LR': 1e-4,
    'adv_LR': 1e-4,
    'batch_size': 128,
    'save_freq': 1,
    'n_sh_coeff': 1 if args.debug else 28,
    'dim_z': 32,
    'num_sites': 6,
}

default_loss_weights = {
    "recon" : 1.0,
    "prior" : 1.0,
    "projection" : 1.0,
    "marg" : 0.01,
    "adv" : 0.0  # 10.0
}

def train(hparams, loss_weights):
    hparams = hparams.copy()
    loss_weights = loss_weights.copy()

    # FIXME add command line arg for dataset path
    train_dataset = TractoinfernoDataset(Path('/home/carl/data/tractoinferno/masked_full'), 'trainset', hparams['n_sh_coeff'], debug=args.debug)
    valid_dataset = TractoinfernoDataset(Path('/home/carl/data/tractoinferno/masked_full'), 'validset', hparams['n_sh_coeff'], debug=args.debug)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hparams['batch_size'],
        shuffle=False,  # Do not shuffle, the dataset has been shuffled. shuffle=True is much slower.
        pin_memory=True,
        num_workers=4
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=hparams['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    center_vox_func = None

    #vec_size = 322
    vec_size = hparams['n_sh_coeff'] * 7

    enc_obj = arch.encoder( vec_size, hparams['dim_z'] )
    dec_obj = arch.decoder( hparams['dim_z'], vec_size, hparams['num_sites'] )
    adv_obj = arch.adv( vec_size, 1 )

    enc_obj.to(device)
    dec_obj.to(device)
    adv_obj.to(device)

    optimizer = torch.optim.Adam(
        list(enc_obj.parameters()) + list(dec_obj.parameters()), lr=hparams['LR']
    )
    adv_optimizer = torch.optim.Adam(adv_obj.parameters(), lr=hparams['adv_LR'])

    comet_experiment = comet_ml.Experiment(project_name='harmon_moyer')
    comet_experiment.log_parameters(hparams)
    comet_experiment.log_parameters(loss_weights, prefix='loss_weight')

    print('Training starts')

    global_step = 0
    global_step_valid = 0

    gen_step_loss_names = [f'loss_{name}' for name in ['recon', 'kl', 'marg', 'adv_g']]
    train_metrics = {name: torchmetrics.MeanMetric(nan_strategy='error') for name in gen_step_loss_names + ['loss_adv_d']}
    valid_metrics = {name: torchmetrics.MeanMetric(nan_strategy='error') for name in gen_step_loss_names + ['loss_adv_d']}

    valid_recon_loss = None

    for epoch in range(hparams['n_epochs']):

        # Training epoch
        for d_idx, batch in enumerate(tqdm(train_loader, desc=f'Train epoch {epoch}')):
            x, c = batch
            x_subj_space = sh_mat = sh_weights = None

            x = x.to(device).type(torch.float32)
            c = c.to(device)

            if False:  # epoch < hparams['burnin_epochs'] or d_idx % (hparams['n_adv_per_enc']+1) > 0:
                adv_optimizer.zero_grad()

                loss = losses.adv_training_step(
                    enc_obj, dec_obj, adv_obj, x, c, num_sites=hparams['num_sites']
                )

                loss.backward(retain_graph=True)
                adv_optimizer.step()

                train_metrics['loss_adv_d'].update(loss.item())
                comet_experiment.log_metric('train_loss_adv_d', loss.item(), step=global_step)
            else:

                optimizer.zero_grad()

                loss, separate_losses = losses.enc_dec_training_step(
                    enc_obj, dec_obj, adv_obj, x, c, loss_weights, hparams['dim_z'], num_sites=hparams['num_sites']
                )

                loss.backward(retain_graph=True)
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
                loss, separate_losses = losses.enc_dec_training_step(
                    enc_obj, dec_obj, adv_obj, x, c, loss_weights, hparams['dim_z'], num_sites=hparams['num_sites']
                )
                for name, l in zip(gen_step_loss_names, separate_losses):
                    try:
                        valid_metrics[name].update(l.item())
                    except RuntimeError:
                        print('While updating metric', name)
                        raise

        # Valid epoch end
        valid_recon_loss = valid_metrics['loss_recon'].compute()
        comet_experiment.log_metrics(
            {name: m.compute() for name, m in valid_metrics.items() if m._update_called},
            epoch=epoch,
            prefix='ep_valid'
        )
        for m in valid_metrics.values():
            m.reset()

        if save_path is not None and epoch % hparams['save_freq'] == 0:
            Path(save_path).mkdir(exist_ok=True)
            torch.save(
                {
                    "enc":enc_obj.state_dict(),
                    "dec":dec_obj.state_dict(),
                    "adv":adv_obj.state_dict()
                },
                f"{save_path}/ckpt_{epoch}.pth"
            )

    return valid_recon_loss


def main():
    study_name = args.optuna_study
    study = optuna.create_study(storage='sqlite:///' + args.optuna_db, study_name=study_name, direction='maximize',
                                load_if_exists=True)

    def objective(trial: optuna.Trial):
        hparams = default_hparams.copy()
        hparams.update({
            'LR': trial.suggest_loguniform('lr', 1e-5, 1e-3),  # 1e-4,
            'adv_LR': trial.suggest_loguniform('lr_adv', 1e-5, 1e-3),
            'dim_z': trial.suggest_categorical('dim_z', [12, 16, 24, 32])
        })
        loss_weights = default_loss_weights.copy()
        loss_weights.update({
            "prior": trial.suggest_loguniform('lw_prior', 1e-1, 1.0),  # 1.0,
            "marg": trial.suggest_loguniform('lw_marg', 1e-3, 1e-1)  #0.01
        })
        valid_recon_loss = train(hparams, loss_weights)
        to_maximize = loss_weights['prior'] + loss_weights['marg']
        to_minimize = valid_recon_loss
        return to_maximize - to_minimize

    study.optimize(objective, n_trials=args.optuna_trials)


if __name__ == '__main__':
    main()
