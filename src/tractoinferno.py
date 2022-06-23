from pathlib import Path

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

    def preprocess(self):
        num_vectors = 0
        vector_size = self.n_sh_coeff * 7

        # Create one array of vectors per volume, save them to disk
        preproc_dir = Path('data') / 'tmp'
        preproc_dir.mkdir(exist_ok=True, parents=True)
        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc='Extract vectors'):
            x, mask = self._get_from_nifti(row['new_id'])
            vectors = extract_vectors_from_volume(x, mask)
            torch.save((vectors, self.site_to_idx[row['site']]), preproc_dir / f'{str(row["new_id"])}.pt')
            num_vectors += len(vectors)

        self.num_vectors = num_vectors

        # Merge all arrays into h5py
        # TODO skip writing .pt on disk? But need to know the number of vectors to create the dataset. Unless we use a
        # resizable h5py dataset.
        with h5py.File(self.h5_filename, mode='w') as f:
            vectors_dset = f.create_dataset("vectors", (num_vectors, vector_size), dtype='float32')
            sites_dset = f.create_dataset("sites", (num_vectors,), dtype='int')

            offset = 0
            for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc='Make hdf5'):
                vectors, site = torch.load(preproc_dir / f'{str(row["new_id"])}.pt')
                n = len(vectors)
                vectors_dset[offset:offset+n] = vectors
                sites_dset[offset:offset+n] = site
                offset += n

    def __init__(self, root_path: Path, set: str, n_sh_coeff: int, force_preprocess=False):
        self.h5_filename = root_path / f'tractoinferno_vectors_{n_sh_coeff}.h5'
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
            print('Preprocessing to' + str(self.h5_filename))
            self.preprocess()

        block_size = 128
        block_size_bytes = block_size * self.n_sh_coeff * 7 * 4
        total_cache_bytes = block_size_bytes * 6
        self.h5_file = h5py.File(self.h5_filename, 'r', rdcc_nbytes=total_cache_bytes)
        self.num_vectors = len(self.h5_file['vectors'])

        self.permutation = self.generate_multiblock_permutation(
            n=self.num_vectors,
            block_size=block_size,
            n_parallel_blocks=4
        )

    @staticmethod
    def generate_multiblock_permutation(n, block_size, n_parallel_blocks):
        n_mblocks = int(np.ceil(n / (block_size * n_parallel_blocks)))
        a = np.arange(n_mblocks * n_parallel_blocks * block_size)
        unused = len(a) - n
        a[-unused:] = -1
        blocks = a.reshape(-1, block_size)
        np.random.shuffle(blocks)
        mblocks = blocks.reshape(-1, n_parallel_blocks * block_size)
        for i in range(3):
            np.random.shuffle(mblocks[i])
        perm = mblocks.flatten()
        perm = perm[perm != -1]
        assert len(perm) == n
        assert max(perm) == n - 1
        return perm

    def __getitem__(self, i):
        i = self.permutation[i]
        return (
            torch.from_numpy(self.h5_file['vectors'][i]),
            torch.tensor(self.h5_file['sites'][i])
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

    TractoinfernoDataset(Path('/home/carl/data/tractoinferno/masked_full'), 'trainset', 2)

