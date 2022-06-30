import numpy as np
import scipy.ndimage
import torch

from src.tractoinferno import extract_vectors_from_volume


def project_vectors_to_site(vectors, target_site, encoder, decoder):
    c = torch.tensor([target_site] * len(vectors))
    z_mu, _ = encoder.forward(vectors)
    recon_vectors = decoder.forward(z_mu, c)
    return recon_vectors


def predict_volume(image, mask, target_site, encoder, decoder):
    # inputs:
    # image             tensor (C, D, H, W)
    # mask              tensor    (D, H, W)
    # target_site       int
    # encoder, decoder  nn.Module   trained modules

    # Erode the mask to see where the neighborhoods are fully wm
    eroded_mask = scipy.ndimage.binary_erosion(mask)

    # Get the indices of non-zero elements
    indices_wm_voxels = eroded_mask.nonzero()

    # Extract vectors for each voxel
    # -> (num_voxels, vector_size)
    vectors = extract_vectors_from_volume(image, mask)
    num_wm_voxels = len(vectors)
    assert num_wm_voxels == len(indices_wm_voxels[0])

    # Predict -- reconstruct voxels into target site
    recon_vectors = project_vectors_to_site(vectors, target_site, encoder, decoder)

    # Extract center voxel from recon_vectors
    num_sh_coef = image.shape[0]
    center_voxels = recon_vectors.reshape(num_wm_voxels, num_sh_coef, 7)[:, :, 3]  # -> (num_voxels, num_sh_coef)

    # Reshape as original shape (C, D, H, W)
    output = np.zeros_like(image)
    for i_chan in range(num_sh_coef):
        output[i_chan][indices_wm_voxels] = center_voxels[i_chan]

    return output
