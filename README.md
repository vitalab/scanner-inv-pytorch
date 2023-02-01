# Harmonization Moyer et al.

This repo is based on code by [Moyer et al.](https://onlinelibrary.wiley.com/doi/10.1002/mrm.28243)
The code was modified to support the TractoInferno dataset. Other changes were
also needed, as decribed in [this document](https://docs.google.com/document/d/1O8dfMagJZ0Acw-0KBTsfLpLOOMRlr2PzcKOcBIYwxrM).

## Setup

Create a virtualenv using `requirements.txt`, and use a virtualenv when
running the scripts.

`requirement_versions.txt` lists versions of packages that work as of this
writing.

## How to launch training

Here is how to launch training with default hyperparameters:

```bash
export PYTHONPATH=$(pwd)  # As required by original Moyer implementation
python src/runner.py <TRACTOINFERNO_DATASET_PATH>
```

Note that the first time this script is run, the dataset is preprocessed to
extract the voxel neighborhoods.

To see what hyperparameters/arguments are available, run
`python src/runner.py  -h`.

## How to evaluate

```bash
python src/posthocadv.py <ZS_FILE>
```

where `ZS_FILE` is a file created at the end of training,
with filename as such: `trainval_zs__{comet_experiment.id}__{epoch}.pth`.