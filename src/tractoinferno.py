from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
import h5py


class TractoinfernoDataset(Dataset):
    num_sites = 6
    set_names = ['trainset', 'validset', 'testset']

    per_coef_mean = np.array([1.73746979e+00,  1.85409542e-02, -1.09420985e-03,  3.89412534e-03, -2.02376191e-02,  4.27327730e-04,  2.23498722e-03, -3.44635831e-04,  1.00409344e-03,  2.43002578e-04,  5.51359786e-04,  9.60882695e-04,  2.90826520e-05,  3.67937580e-04, -1.05259285e-04, -5.80608976e-05, -1.20373130e-04,  2.14313077e-05, -2.46529380e-04,  4.79140981e-05, -2.40324925e-06,  7.33948618e-05, -1.06283391e-04, -2.59366807e-05, -1.57749411e-04, -1.10661149e-05, -7.32169474e-06, -1.32354195e-04])
    per_coef_std = np.array([0.19366595, 0.13481042, 0.12510094, 0.13810773, 0.12710623, 0.11857963, 0.01354882, 0.01217443, 0.01309844, 0.012926  , 0.01303048, 0.01326259, 0.01199164, 0.01236033, 0.0126209 , 0.00339288, 0.00331109, 0.00333396, 0.00345094, 0.0033553 , 0.00338018, 0.00340034, 0.00342043, 0.00337105, 0.00337168, 0.00328506, 0.0033479 , 0.00332702])

    vec_mean = np.repeat(per_coef_mean, 7)
    vec_std = np.repeat(per_coef_std, 7)

    def _get_from_nifti(self, volume_id):
        def load_nifti_as_tensor(nifti_file):
            nifti = nib.load(nifti_file)
            npy_volume = nifti.get_fdata('unchanged', np.float32)
            return torch.from_numpy(npy_volume).to(torch.float32)

        image = load_nifti_as_tensor(self.subset_path / volume_id / f'sh/{volume_id}__dwi_sh_wm.nii.gz')
        image = torch.movedim(image, -1, 0)  # Move SH dim to index 0
        image = image[:self.n_sh_coeff]

        mask = load_nifti_as_tensor(self.subset_path / volume_id / f'mask/{volume_id}__mask_wm.nii.gz')

        return image, mask

    def preprocess(self, debug_dataset=False):
        # debug_dataset=True will create a smaller dataset for debugging purposes

        num_vectors = 0
        vector_size = self.n_sh_coeff * 7

        # Create one array of vectors per volume, save them to disk
        preproc_dir = Path('data') / 'tmp'
        preproc_dir.mkdir(exist_ok=True, parents=True)
        for f in preproc_dir.glob('*.pt'):
            f.unlink()

        keep = 10000
        rng = np.random.default_rng()
        n_per_site = {s: 0 for s in range(6)}
        skipped = []
        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc='Extract vectors'):
            if debug_dataset:
                if n_per_site[self.site_to_idx[row['site']]] == 2:
                    skipped.append(i)
                    continue
                n_per_site[self.site_to_idx[row['site']]] += 1

            x, mask = self._get_from_nifti(row['new_id'])
            vectors = extract_vectors_from_volume(x, mask)
            vectors = rng.choice(vectors, size=keep)  # Sample a subset of the voxels
            torch.save((vectors, self.site_to_idx[row['site']]), preproc_dir / f'{str(row["new_id"])}.pt')
            num_vectors += len(vectors)

        self.num_vectors = num_vectors

        shuffled_indices = np.random.default_rng().permutation(num_vectors)

        # Merge all arrays into h5py
        with h5py.File(self.h5_filename, mode='w') as f:
            vectors_dset = f.create_dataset("vectors", (num_vectors, vector_size), dtype='float32')
            sites_dset = f.create_dataset("sites", (num_vectors,), dtype='int')

            offset = 0
            for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc='Make hdf5'):
                if debug_dataset and i in skipped:
                    continue
                vectors, site = torch.load(preproc_dir / f'{str(row["new_id"])}.pt')
                num = len(vectors)

                indices = np.sort(shuffled_indices[offset:offset+num])
                vectors_dset[indices] = vectors
                sites_dset[indices] = site
                offset += num

    def __init__(self, root_path: Path, set: str, n_sh_coeff: int, force_preprocess=False, debug=False):
        basename = f'tractoinferno_vectors_{set}_{n_sh_coeff}'
        if debug:
            basename += '_debug'
        self.h5_filename = root_path / (basename + '.h5')
        self.n_sh_coeff = n_sh_coeff
        self.subset_path = root_path / set
        full_df = pd.read_csv(root_path / 'metadata.csv')
        self.df = full_df[full_df['dataset'] == set]
        self.num_vectors = None

        self.idx_to_site = self.df['site'].unique().tolist()
        assert len(self.idx_to_site) == TractoinfernoDataset.num_sites
        self.site_to_idx = {s: i for i, s in enumerate(self.idx_to_site)}

        if not force_preprocess and self.h5_filename.exists():
            print('Loading from ' + str(self.h5_filename))
        else:
            print('Preprocessing to ' + str(self.h5_filename))
            self.preprocess(debug_dataset=debug)

        self.h5_file = h5py.File(self.h5_filename, 'r')
        self.num_vectors = len(self.h5_file['vectors'])

    def __getitem__(self, i):
        x = self.h5_file['vectors'][i]
        x = (x - self.vec_mean) / self.vec_std  # TODO perform normalization once during preprocessing?
        y = self.h5_file['sites'][i]
        return (
            torch.from_numpy(x),
            torch.tensor(y)
        )

    def __len__(self):
        return self.num_vectors


def extract_vectors_from_volume(volume, mask):
    # volume (C, D, H, W)
    # mask      (D, H, W)

    vol_and_mask = np.concatenate([volume, mask[None, ...]], axis=0)  # shape: (C + 1, D, H, W)

    # window_shape: (C+1, 3, 3, 3)
    window_shape = (volume.shape[0] + 1, 3, 3, 3)
    windows = np.lib.stride_tricks.sliding_window_view(vol_and_mask, window_shape)

    # Separate windowed volume and mask
    windows_vol = windows[..., :-1, :, :, :]
    windows_mask = windows[..., -1, :, :, :]

    # Flatten neighborhoods (but don't flatten completely)
    windows_vol = windows_vol.reshape(-1, volume.shape[0], 27)  # (num_vectors, C, 27)
    windows_mask = windows_mask.reshape(-1, 27)  # (num_vectors, 27)

    # Keep only center voxel and direct neighbors
    valid_neighborhood = [4, 10, 12, 13, 14, 16, 22]
    windows_vol = windows_vol[:, :, valid_neighborhood]
    windows_mask = windows_mask[:, valid_neighborhood]

    # Keep only voxels for which the full neighborhood is in white-matter
    all_in_wm = windows_mask.sum(axis=1) == 7
    windows_vol = windows_vol[all_in_wm]

    # Flatten
    vectors = windows_vol.reshape(-1, volume.shape[0] * 7)

    return vectors


def test_extract_vectors_from_volume():
    # Signal:
    # 0160
    # 2345
    # 9780
    # 0100
    #
    # Mask:
    # 0110
    # 1111
    # 0111
    # 0010
    #
    # We expect 3 vectors in the output

    volume_2d = np.array([[0, 1, 6, 0], [2, 3, 4, 5], [9, 7, 8, 0], [0, 1, 0, 0]])
    mask_2d = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 0]])
    expected_vectors = np.array([[0, 1, 2, 3, 4, 7, 0], [0, 6, 3, 4, 5, 8, 0], [0, 4, 7, 8, 0, 0, 0]])

    volume = np.stack([np.zeros_like(volume_2d), volume_2d, np.zeros_like(volume_2d)])
    volume = np.stack([volume, volume])  # Add a channels axis
    expected_vectors = np.stack([expected_vectors, expected_vectors], axis=1).reshape(-1, 14)
    mask = np.stack([mask_2d, mask_2d, mask_2d])

    vectors = extract_vectors_from_volume(volume, mask)

    assert np.all(vectors == expected_vectors)
    print('test_extract_vectors_from_volume : Success.')


if __name__ == "__main__":
    test_extract_vectors_from_volume()

    TractoinfernoDataset(Path('/home/carl/data/tractoinferno/masked_full'), 'trainset', 28, force_preprocess=True)

