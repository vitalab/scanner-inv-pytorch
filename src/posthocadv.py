import argparse
from pathlib import Path

import comet_ml  # Comet refuses to work if not imported before dl libs
import numpy
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import CometLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy


class SitePredictorSystem(pl.LightningModule):
    def __init__(self, code_dim, num_sites, weight_decay=0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(code_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_sites)
        )
        self.weight_decay = weight_decay

        self.accu_train = Accuracy(task='multiclass', num_classes=num_sites)
        self.accu_valid = Accuracy(task='multiclass', num_classes=num_sites)

        self.best_accu_train = 0
        self.best_accu_valid = 0

    def training_step(self, batch, batch_idx):
        z, site_label = batch
        z = z.float()
        s_logits = self.model(z)
        loss = F.cross_entropy(s_logits, site_label)

        self.accu_train.update(s_logits, site_label)
        self.best_accu_train = max(self.best_accu_train, self.accu_train.compute().item())
        self.log('accu_train', self.accu_train, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        z, site_label = batch
        z = z.float()
        s_logits = self.model(z)
        loss = F.cross_entropy(s_logits, site_label)
        self.log('val_loss', loss)

        self.accu_valid.update(s_logits, site_label)
        self.best_accu_valid = max(self.best_accu_valid, self.accu_valid.compute().item())
        self.log('accu_valid', self.accu_valid, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.weight_decay)
        return optimizer


def load_posthoc_dataset(preds):
    data_z, data_labels = [], []
    for pred_batch in preds:
        z, s = pred_batch
        if isinstance(z, numpy.ndarray):
            z = torch.from_numpy(z)
        data_z.append(z)
        data_labels.append(s)
    data_z = torch.cat(data_z)
    data_labels = torch.cat(data_labels)

    return TensorDataset(data_z, data_labels)


def train_posthoc_adv(zs_file_path: Path, args):
    device = {0: 'cpu', 1: 'cuda'}[int(args.gpus)]

    trained_cvae_model_id = str(zs_file_path.name).split('__')[1]

    # Prepare the datasets for training the post-hoc adversary
    # Structure of z,s: {'train_zs': train_zs, 'valid_zs': valid_zs}, where *_zs is a list of pairs of batches

    zs_data = torch.load(zs_file_path, map_location=device)
    code_dim = zs_data['train_zs'][0][0].shape[1]
    train_posthoc_dataset = load_posthoc_dataset(zs_data['train_zs'])
    valid_posthoc_dataset = load_posthoc_dataset(zs_data['valid_zs'])

    train_dataloader = DataLoader(train_posthoc_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_dataloader = DataLoader(valid_posthoc_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # init adversary model
    adversary = SitePredictorSystem(code_dim, args.n_sites, args.weight_decay)

    # Logger
    logger = CometLogger(project_name='harmon-moyer', display_summary_level=0,
                         disabled=args.no_comet)  # save_dir='.',
    logger.experiment.add_tag('eval')
    logger.log_hyperparams({'trained_cvae_model_id': trained_cvae_model_id})
    logger.log_hyperparams(args)

    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        logger=logger,
        default_root_dir=args.root_dir
    )

    # Train the model ⚡
    trainer.fit(adversary, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    # Log eval metrics in the comet experiment of the cvae training
    existing_exp = comet_ml.ExistingExperiment(experiment_key=trained_cvae_model_id)
    existing_exp.log_metrics({
        'best_accu_train': adversary.best_accu_train,
        'best_accu_valid': adversary.best_accu_valid
    }, prefix='posthoc')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zs_file', type=Path)
    ap.add_argument('--dataset', default='tractoinferno')
    ap.add_argument('--gpus', default=0)
    ap.add_argument('-e', '--max_epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no_comet', action='store_true')
    ap.add_argument('--root_dir', type=str)
    args = ap.parse_args()

    if args.dataset == 'tractoinferno':
        args.n_sites = 6
    else:
        raise NotImplementedError

    train_posthoc_adv(args.zs_file, args)


if __name__ == '__main__':
    main()
