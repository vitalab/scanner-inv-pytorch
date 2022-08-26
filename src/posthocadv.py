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
    def __init__(self, code_dim, num_sites):
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

        self.accu_train = Accuracy()
        self.accu_valid = Accuracy()

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
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
    adversary = SitePredictorSystem(code_dim, args.n_sites)

    # Logger
    logger = CometLogger(project_name='harmon-moyer', display_summary_level=0,
                         disabled=args.no_comet)  # save_dir='.',
    logger.experiment.add_tag('eval')
    logger.log_hyperparams({'trained_cvae_model_id': trained_cvae_model_id})
    logger.log_hyperparams(args)

    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=100,
        logger=logger,
        default_root_dir=args.root_dir
    )

    # Train the model ⚡
    trainer.fit(adversary, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    # Log eval metrics in the comet experiment of the cvae training
    try:
        existing_exp = comet_ml.ExistingExperiment(experiment_key=trained_cvae_model_id)
    except ValueError:
        # TODO remove this eventually
        # Assuming ValueError: Invalid experiment key
        # If old code was used, the filename contains the name instead of the ID.
        name_to_id = {
            'genetic_seasoning_4914': 'ee6d77c3d3174332aa2ec58863fdc64f',
            'integral_thrush_9169': 'bb5356f242c94f8f80648c27aef2bfac',
            'keen_arch_9585': '44de53fe68d44202bf041813df90f09b',
            'marked_antelope_4111': 'e8c89edcd118437cbe2faecd7d79fbd8',
            'large_tropics_9452': 'f5f279b7fc3b4e918c261c76dfd77609',
            'open_berm_2043': '0fabc4156f52466196fa9151e994be54',
            'content_crypt_5612': '98935da09de2486ba0c4d347c7018a8a',
            'plum_valley_5854': '15e3bd4a521b42a28d966b3c63b3ae05',
            'advanced_monkey_7523': 'd1f6739d52494a4e9dae44bdd5ce34b8',
            'administrative_dragon_6353': 'ba409b916fcf498fa839388f11061cfe',
            'tomato_rice_8561': 'b4a5a3f6219b402389bcf4bee1b1e779',
            'symbolic_goldfish_6434': '340e030eebed42e18cd22d257ca454b0',
            'royal_clam_3891': '4c3abf848d0d4c0191c740232844a81e',
            'concrete_wildebeest_4570': 'aae78467a6bf4d199304ea6597cbe4d9',
            'specified_salamander_2130': 'b784b437968d4981ac321e7391cca1c6',
            'exact_primate_8076': '2e3e29c2119c4d5188f93d767958e0dd',
            'adverse_liability_9915': '5375683d433545a0b64e092cc87697a9',
            'worrying_planarian_7313': '2e9863603d384df696d42fba65ab7120',
            'neutral_cellulose_4717': '50dcac07676a42e79d16c3176ae4874a',
            'ok_cream_571': 'da1c5e255a294ecaa094e70c8ee61d9b',
            'marked_gooseberry_3866': '7133a633b36f4c6ba08c551f04e65f9c',
            'living_catshark_225': '81506ab5ce6a463f8d080f96905a2487',
            'annual_weasel_1913': '9fe74778668c49169ffbd1e13a8832b2',
            'white_relativity_5855': '8ef2282f72e54ff382f1c4c181b5548a',
        }
        trained_cvae_model_id = name_to_id[trained_cvae_model_id]
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
