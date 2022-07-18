

import torch
import torch.nn.functional as F


def kl_loss_gaussians( mu, log_sigma_sq, eps=1e-8):

    kl_loss_term1 = -(mu * mu)
    kl_loss_term2 = -log_sigma_sq.exp()
    kl_loss_term3 = 1 + log_sigma_sq

    kl_loss = -0.5 * (kl_loss_term1 + kl_loss_term2 + kl_loss_term3).sum(axis=1)

    #TODO: axes check
    return kl_loss.mean()


def all_pairs_gaussian_kl(mu, log_sigma_sq, dim_z):

    sigma_sq = log_sigma_sq.exp()
    sigma_sq_inv = 1.0 / sigma_sq

    first_term = torch.matmul(sigma_sq, sigma_sq_inv.transpose(0,1))

    r = torch.matmul( mu * mu, sigma_sq_inv.squeeze().transpose(0,1) )
    r2 = mu * mu * sigma_sq_inv.squeeze()
    r2 = r2.sum(axis=1)

    second_term = 2*torch.matmul(mu, (mu*sigma_sq_inv.squeeze()).transpose(0,1))
    second_term = r - second_term + r2

    #this is det(Sigma), because we don't have off diag elements
    det_sigma = log_sigma_sq.sum(axis=1)
    # from each det(sigma) we need to subtract every other det(Sigma)
    third_term = torch.unsqueeze(det_sigma, dim=1) - torch.unsqueeze(det_sigma, dim=1).transpose(0,1)

    return 0.5 * ( first_term + second_term + third_term - dim_z )

#
# enc, dec should be in training mode
# adv should be in eval mode
def enc_dec_training_step( encoder, decoder, adv, x, c, loss_weights, dim_z, num_sites):
    z_mu, z_log_sigma_sq = encoder.forward(x)

    kl_loss = kl_loss_gaussians( z_mu, z_log_sigma_sq )

    std = torch.exp(0.5 * z_log_sigma_sq)
    eps = torch.randn_like(std)
    z = eps * std + z_mu
    x_recon = decoder.forward( z, F.one_hot(c, num_classes=num_sites) )

    recon_loss = F.mse_loss( x_recon, x )

    #TODO: projection loss

    # 
    # KL[q(z|x) | q(z)]
    marg_loss = all_pairs_gaussian_kl( z_mu, z_log_sigma_sq, dim_z ).mean()
    #marg_loss = 0

    adv_likelihood = adv.forward(x_recon)
    adv_gt = torch.zeros_like(adv_likelihood)
    #adv_loss = torch.nn.BCELoss(adv_likelihood, adv_gt)
    adv_loss = F.binary_cross_entropy( adv_likelihood, adv_gt )

    recon_loss *= loss_weights['recon']
    kl_loss *= loss_weights['prior']
    marg_loss *= loss_weights['marg']
    adv_loss *= loss_weights['adv_g']
    loss = recon_loss + kl_loss + marg_loss - adv_loss

    return loss, (recon_loss, kl_loss, marg_loss, adv_loss), z_mu, x_recon


def adv_training_step( encoder, decoder, adv, x, c, num_sites ):
    N_half = x.size()[0] // 2
    x1 = x[:N_half]
    #c1 = c[:N_half] #unused
    x2 = x[N_half:]
    c2 = c[N_half:]

    z_mu, z_log_sigma_sq = encoder.forward(x2)
    std = torch.exp(0.5 * z_log_sigma_sq)
    eps = torch.randn_like(std)
    z = eps * std + z_mu
    x_recon = decoder.forward(z, F.one_hot(c2, num_classes=num_sites) )

    labels_pred = torch.cat((
        adv.forward(x1),
        adv.forward(x_recon)
    ))

    labels = torch.cat(
        ( torch.ones(x1.size()[0]), torch.zeros(x_recon.size()[0]) )
    ).to(labels_pred.device)

    #loss = torch.nn.BCELoss( labels_pred, labels )
    loss = F.binary_cross_entropy( labels_pred.squeeze(), labels.squeeze() )

    return loss 







