from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm


class TractoinfernoDataset(Dataset):
    num_sites = 6
    set_names = ['trainset', 'validset', 'testset']

    def _get_from_nifti(self, index, volume_id):
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
        all_vectors = []
        all_sites = []

        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc='Preprocess'):
            x, mask = self._get_from_nifti(None, row['new_id'])
            vectors = extract_vectors_from_volume(x, mask)
            vectors = torch.from_numpy(vectors)
            sites = torch.tensor([self.site_to_idx[row['site']]] * len(vectors))
            all_vectors.append(vectors)
            all_sites.append(sites)

            torch.save((vectors, self.site_to_idx[row['site']]), f'preproc_{str(row["new_id"])}.pt')  # TODO remove this

        all_vectors = torch.cat(all_vectors)
        all_sites = torch.cat(all_sites)

        return all_vectors, all_sites

    def __init__(self, root_path: Path, set: str, n_sh_coeff: int, load_cached=True):
        self.n_sh_coeff = n_sh_coeff
        self.subset_path = root_path / set
        full_df = pd.read_csv(root_path / 'metadata.csv')
        self.df = full_df[full_df['dataset'] == set]

        self.idx_to_site = self.df['site'].unique().tolist()
        assert len(self.idx_to_site) == TractoinfernoDataset.num_sites
        self.site_to_idx = {s: i for i, s in enumerate(self.idx_to_site)}

        cache_file = Path('tractoinferno.pt')
        if load_cached and cache_file.exists():
            data = torch.load(cache_file)
        else:
            data = self.preprocess()
            torch.save(data, cache_file)

        self.vectors, self.sites = data


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

