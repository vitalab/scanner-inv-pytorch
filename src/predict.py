import argparse
from pathlib import Path

import numpy as np
import scipy.ndimage
import torch
from torch.nn.functional import one_hot
import nibabel as nib

from src.tractoinferno import extract_vectors_from_volume
from src.arch import encoder as Encoder, decoder as Decoder
from src.tractoinferno import TractoinfernoDataset


class Config:
    num_sites = 6


def project_vectors_to_site(vectors, target_site, encoder, decoder):
    vectors = torch.from_numpy(vectors)
    c = torch.tensor([target_site] * len(vectors))

    with torch.no_grad():
        z_mu, _ = encoder.forward(vectors)
        recon_vectors = decoder.forward(z_mu, one_hot(c, num_classes=Config.num_sites))
    return recon_vectors.numpy()


def predict_volume(image, mask, target_site, encoder, decoder):
    # inputs:
    # image             tensor (C, D, H, W)
    # mask              tensor    (D, H, W)
    # target_site       int
    # encoder, decoder  nn.Module   trained modules

    print("  Preprocessing")

    # Erode the mask to see where the neighborhoods are fully wm
    eroded_mask = scipy.ndimage.binary_erosion(mask)

    # Get the indices of non-zero elements
    indices_wm_voxels = eroded_mask.nonzero()

    # Extract vectors for each voxel
    # -> (num_voxels, vector_size)
    vectors = extract_vectors_from_volume(image, mask)
    num_wm_voxels = len(vectors)
    assert num_wm_voxels == len(indices_wm_voxels[0])

    print("  Running model")
    # Predict -- reconstruct voxels into target site
    recon_vectors = project_vectors_to_site(vectors, target_site, encoder, decoder)

    print("  Reshaping")
    # Extract center voxel from recon_vectors
    num_sh_coef = image.shape[0]
    center_voxels = recon_vectors.reshape(num_wm_voxels, num_sh_coef, 7)[:, :, 3]  # -> (num_voxels, num_sh_coef)

    # Reshape as original shape (C, D, H, W)
    output = np.zeros_like(image)
    for i_chan in range(num_sh_coef):
        output[i_chan][indices_wm_voxels] = center_voxels[:, i_chan]

    return output


def load_nifti_as_tensor(nifti_file):
    nifti = nib.load(nifti_file)
    npy_volume = nifti.get_fdata('unchanged', np.float32)
    return torch.from_numpy(npy_volume).to(torch.float32), nifti


def main():
    """Harmonize on a volume by projecting it to a target site"""

    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='Nifti file of the image')
    ap.add_argument('--mask', required=True, help='Nifti file of the white matter mask')
    ap.add_argument('--model_weights', required=True)
    ap.add_argument('--target_site', type=int, default=0)
    ap.add_argument('--vector_size', type=int, default=28 * 7)
    ap.add_argument('--dim_z', type=int, default=32)
    ap.add_argument('--num_sites', type=int, default=Config.num_sites)
    args = ap.parse_args()

    Config.num_sites = args.num_sites

    # Load model
    print("Loading model")
    checkpoint = torch.load(args.model_weights, map_location='cpu')
    encoder = Encoder(args.vector_size, args.dim_z)
    decoder = Decoder(args.dim_z, args.vector_size, args.num_sites)
    encoder.load_state_dict(checkpoint['enc'])
    decoder.load_state_dict(checkpoint['dec'])

    # Load image and mask
    print("Loading data")
    image, nifti_object = load_nifti_as_tensor(args.image)
    image = torch.movedim(image, -1, 0)  # Move SH dim to the front
    mask, _ = load_nifti_as_tensor(args.mask)

    # Predict
    print("Predicting")
    recon_image = predict_volume(image, mask, args.target_site, encoder, decoder)
    recon_image = np.moveaxis(recon_image, 0, -1)  # Move SH dim to the back

    print("Saving output")
    # Make filename
    version = Path(args.model_weights).stem
    orig_base_name = Path(args.image).name
    orig_stem = orig_base_name[:-7] if orig_base_name[-7:] == '.nii.gz' else orig_base_name[:-4]
    out_filename = Path(args.image).parent / f'{orig_stem}_harmonizedTo{args.target_site}_{version}.nii'

    # Save output
    out_nifti = nib.Nifti1Pair(recon_image, nifti_object.affine, nifti_object.header)
    nib.nifti1.save(out_nifti, str(out_filename))
    print('Saved prediction as', out_filename)


if __name__ == '__main__':
    main()
