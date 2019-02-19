"""
Attend, Infer, Repeat:
Fast Scene Understanding with Generative Models
https://arxiv.org/pdf/1603.08575v2.pdf
"""

import os
import argparse
import pprint
import time
from tqdm import tqdm

import numpy as np
from observations import multi_mnist

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()

# actions
parser.add_argument('--train', action='store_true', help='Train a model.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate a model.')
parser.add_argument('--generate', action='store_true', help='Generate samples from a model.')
parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
parser.add_argument('--data_dir', default='./data/', help='Location of datasets.')
parser.add_argument('--output_dir', default='./results/{}'.format(os.path.splitext(__file__)[0]))
parser.add_argument('--seed', type=int, default=2182019, help='Random seed to use.')
parser.add_argument('--cuda', type=int, default=None, help='Which cuda device to use')
parser.add_argument('--verbose', '-v', action='count', help='Verbose mode; send gradient stats to tensorboard.')
# model params
parser.add_argument('--image_dims', type=tuple, default=(1,50,50), help='Dimensions of a single datapoint (e.g. (1,50,50) for multi MNIST).')
parser.add_argument('--z_what_size', type=int, default=50, help='Size of the z_what latent representation.')
parser.add_argument('--z_where_size', type=int, default=3, help='Size of the z_where latent representation e.g. dim=3 for (s, tx, ty) affine parametrization.')
parser.add_argument('--z_pres_size', type=int, default=1, help='Size of the z_pres latent representation, e.g. dim=1 for the probability of occurence of an object.')
parser.add_argument('--enc_dec_size', type=int, default=200, help='Size of the encoder and decoder hidden layers.')
parser.add_argument('--lstm_size', type=int, default=256, help='Size of the LSTM hidden layer for AIR.')
parser.add_argument('--baseline_lstm_size', type=int, default=256, help='Size of the LSTM hidden layer for the gradient baseline estimator.')
parser.add_argument('--attn_window_size', type=int, default=28, help='Size of the attention window of the decoder.')
parser.add_argument('--max_steps', type=int, default=3, help='Maximum number of objects per image to sample a binomial from.')
parser.add_argument('--likelihood_sigma', type=float, default=0.3, help='Sigma parameter for the likelihood function (a Normal distribution).')
parser.add_argument('--z_pres_prior_success_prob', type=float, default=0.75, help='Prior probability of success for the num objects per image prior.')
parser.add_argument('--z_pres_anneal_start_step', type=int, default=1000, help='Start step to begin annealing the num objects per image prior.')
parser.add_argument('--z_pres_anneal_end_step', type=int, default=100000, help='End step to stop annealing the num objects per image prior.')
parser.add_argument('--z_pres_anneal_start_value', type=float, default=0.99, help='Initial probability of success for the num objects per image prior.')
parser.add_argument('--z_pres_anneal_end_value', type=float, default=1e-5, help='Final probility of successs value for the num objects per image prior.')
parser.add_argument('--z_pres_init_encoder_bias', type=float, default=2., help='Add bias to the initialization of the z_pres encoder.')
parser.add_argument('--decoder_bias', type=float, default=-2., help='Add preactivation bias to decoder.')
# training params
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--start_epoch', default=0, help='Starting epoch (for logging; to be overwritten when restoring file.')
parser.add_argument('--model_lr', type=float, default=1e-4, help='Learning rate for AIR.')
parser.add_argument('--baseline_lr', type=float, default=1e-3, help='Learning rate for the gradient baseline estimator.')
parser.add_argument('--log_interval', type=int, default=100, help='Write loss and parameter stats to tensorboard.')
parser.add_argument('--eval_interval', type=int, default=10, help='Number of epochs to eval model and save checkpoint.')
parser.add_argument('--mini_data_size', type=int, default=None, help='Train only on this number of datapoints.')

# --------------------
# Data
# --------------------

class MultiMNIST(Dataset):
    def __init__(self, root, training=True, download=True, max_digits=2, canvas_size=50, seed=42, mini_data_size=None):
        self.root = os.path.expanduser(root)

        # check if multi mnist already compiled
        self.multi_mnist_filename = 'multi_mnist_{}_{}_{}'.format(max_digits, canvas_size, seed)

        if not self._check_processed_exists():
            if self._check_raw_exists():
                # process into pt file
                data = np.load(os.path.join(self.root, 'raw', self.multi_mnist_filename + '.npz'))
                train_data, train_labels, test_data, test_labels = [data[f] for f in data.files]
                self._process_and_save(train_data, train_labels, test_data, test_labels)
            else:
                if not download:
                    raise RuntimeError('Dataset not found. Use download=True to download it.')
                else:
                    (train_data, train_labels), (test_data, test_labels) = multi_mnist(root, max_digits, canvas_size, seed)
                    self._process_and_save(train_data, train_labels, test_data, test_labels)
        else:
            data = torch.load(os.path.join(self.root, 'processed', self.multi_mnist_filename + '.pt'))
            self.train_data, self.train_labels, self.test_data, self.test_labels = \
                    data['train_data'], data['train_labels'], data['test_data'], data['test_labels']

        if training:
            self.x, self.y = self.train_data, self.train_labels
        else:
            self.x, self.y = self.test_data, self.test_labels

        if mini_data_size != None:
            self.x = self.x[:mini_data_size]
            self.y = self.y[:mini_data_size]

    def __getitem__(self, idx):
        return self.x[idx].unsqueeze(0), self.y[idx]

    def __len__(self):
        return len(self.x)

    def _check_processed_exists(self):
        return os.path.exists(os.path.join(self.root, 'processed', self.multi_mnist_filename + '.pt'))

    def _check_raw_exists(self):
        return os.path.exists(os.path.join(self.root, 'raw', self.multi_mnist_filename + '.npz'))

    def _make_label_tensor(self, label_arr):
        out = torch.zeros(10)
        for l in label_arr:
            out[l] += 1
        return out

    def _process_and_save(self, train_data, train_labels, test_data, test_labels):
        self.train_data = torch.from_numpy(train_data).float() / 255
        self.train_labels = torch.stack([self._make_label_tensor(label) for label in train_labels])
        self.test_data = torch.from_numpy(test_data).float() / 255
        self.test_labels = torch.stack([self._make_label_tensor(label) for label in test_labels])
        # check folder exists
        if not os.path.exists(os.path.join(self.root, 'processed')):
            os.makedirs(os.path.join(self.root, 'processed'))
        with open(os.path.join(self.root, 'processed', self.multi_mnist_filename + '.pt'), 'wb') as f:
            torch.save({'train_data': self.train_data,
                        'train_labels': self.train_labels,
                        'test_data': self.test_data,
                        'test_labels': self.test_labels},
                        f)

def fetch_dataloaders(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.device.type is 'cuda' else {}
    dataset = MultiMNIST(root=args.data_dir, training=True, mini_data_size=args.mini_data_size)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    dataset = MultiMNIST(root=args.data_dir, training=False if args.mini_data_size is None else True, mini_data_size=args.mini_data_size)
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, **kwargs)
    return train_dataloader, test_dataloader


# --------------------
# Model helper functions -- spatial tranformer
# --------------------

def stn(image, z_where, out_dims, inverse=False, box_attn_window_color=None):
    """ spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True

    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]

    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T

    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    """

    if box_attn_window_color is not None:
        # draw a box around the attention window by overwriting the boundary pixels in the given color channel
        with torch.no_grad():
            box = torch.zeros_like(image.expand(-1,3,-1,-1))
            c = box_attn_window_color % 3  # write the color bbox in channel c, as model time steps 
            box[:,c,:,0] = 1
            box[:,c,:,-1] = 1
            box[:,c,0,:] = 1
            box[:,c,-1,:] = 1
            # add box to image and clap at 1 if overlap
            image = torch.clamp(image + box, 0, 1)

    # 1. construct 2x3 affine matrix for each datapoint in the minibatch
    theta = torch.zeros(2,3).repeat(image.shape[0], 1, 1).to(image.device)
    # set scaling
    theta[:, 0, 0] = theta[:, 1, 1] = z_where[:,0] if not inverse else 1 / (z_where[:,0] + 1e-9)
    # set translation
    theta[:, :, -1] = z_where[:, 1:] if not inverse else - z_where[:,1:] / (z_where[:,0].view(-1,1) + 1e-9)
    # 2. construct sampling grid
    grid = F.affine_grid(theta, torch.Size(out_dims))
    # 3. sample image from grid
    return F.grid_sample(image, grid)


# --------------------
# Model helper functions -- distribution manupulations
# --------------------

def compute_geometric_from_bernoulli(obj_probs):
    """ compute a normalized truncated geometric distribution from a table of bernoulli probs
    args
        obj_probs -- tensor of shape (N, max_steps) of Bernoulli success probabilities.
    """
    cum_succ_probs = obj_probs.cumprod(1)
    fail_probs = 1 - obj_probs
    geom = torch.cat([fail_probs[:,:1], fail_probs[:,1:] * cum_succ_probs[:,:-1], cum_succ_probs[:,-1:]], dim=1)
    return geom / geom.sum(1, True)

def compute_z_pres_kl(q_z_pres_geom, p_z_pres, writer=None):
    """ compute kl divergence between truncated geom prior and tabular geom posterior
    args
        p_z_pres -- torch.distributions.Geometric object
        q_z_pres_geom -- torch tensor of shape (N, max_steps + 1) of a normalized geometric pdf
    """
    # compute normalized truncated geometric
    p_z_pres_log_probs = p_z_pres.log_prob(torch.arange(q_z_pres_geom.shape[1], dtype=torch.float, device=q_z_pres_geom.device))
    p_z_pres_normed_log_probs = p_z_pres_log_probs - p_z_pres_log_probs.logsumexp(dim=0)

    kl = q_z_pres_geom * (torch.log(q_z_pres_geom + 1e-8) - p_z_pres_normed_log_probs.expand_as(q_z_pres_geom))
    return kl

def anneal_z_pres_prob(prob, step, args):
    if args.z_pres_anneal_start_step < step < args.z_pres_anneal_end_step:
        slope = (args.z_pres_anneal_end_value - args.z_pres_anneal_start_value) / (args.z_pres_anneal_end_step - args.z_pres_anneal_start_step)
        prob = torch.tensor(args.z_pres_anneal_start_value + slope * (step - args.z_pres_anneal_start_step), device=prob.device)
    return prob


# --------------------
# Model
# --------------------

class AIR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.debug = False
        # record dims
        self.C, self.H, self.W = args.image_dims
        self.A = args.attn_window_size
        x_size = self.C * self.H * self.W
        self.lstm_size = args.lstm_size
        self.baseline_lstm_size = args.baseline_lstm_size
        self.z_what_size = args.z_what_size
        self.z_where_size = args.z_where_size
        self.max_steps = args.max_steps

        # --------------------
        # p model -- cf AIR paper section 2
        # --------------------

        # latent variable priors
        # z_pres  ~ Ber(p) Geom(rho) discrete representation for the presence of a scene object
        # z_where ~ N(mu, scale); continuous 3-dim variable for pose (position and scale)
        # z_what  ~ N(0,1); continuous representation for shape
        self.register_buffer('z_pres_prior', torch.tensor(args.z_pres_prior_success_prob))  # prior used for generation
        self.register_buffer('z_pres_prob', torch.tensor(args.z_pres_anneal_start_value))   # `current value` used for training and annealing
        self.register_buffer('z_what_mean', torch.zeros(args.z_what_size))
        self.register_buffer('z_what_scale', torch.ones(args.z_what_size))
        self.register_buffer('z_where_mean', torch.tensor([0.3, 0., 0.]))
        self.register_buffer('z_where_scale', torch.tensor([0.1, 1., 1.]))

        # likelihood = N(mu, sigma)
        self.register_buffer('likelihood_sigma', torch.tensor(args.likelihood_sigma))

        # likelihood p(x|n,z) of the data given the latents
        self.decoder = nn.Sequential(nn.Linear(args.z_what_size, args.enc_dec_size),
                                     nn.ReLU(True),
                                     nn.Linear(args.enc_dec_size, self.C * self.A ** 2))
        self.decoder_bias = args.decoder_bias  # otherwise initial samples are heavily penalized by likelihood (cf Pyro implementation)

        # --------------------
        # q model for approximating the posterior -- cf AIR paper section 2.1
        # --------------------

        # encoder
        #   rnn encodes the latents z_1:t over the number of steps where z_pres indicates presence of an object
        #   q_z_pres encodes whether there is an object present in the image; q_z_pres = Bernoulli
        #   q_z_what encodes the attention window; q_z_what = Normal(mu, sigma)
        #   q_z_where encodes the affine transform of of the image > attn_window; q_z_where = Normal(0, cov) of dim = 3 for [scale, tx, ty]
        self.encoder = nn.ModuleDict({
            'rnn':      nn.LSTMCell(x_size + args.z_where_size + args.z_what_size + args.z_pres_size, args.lstm_size),
            'z_pres':   nn.Linear(args.lstm_size, 1),
            'z_what':   nn.Sequential(nn.Linear(self.A ** 2 , args.enc_dec_size),
                                      nn.ReLU(True),
                                      nn.Linear(args.enc_dec_size, 2 * args.z_what_size)),
            'z_where':  nn.Linear(args.lstm_size, 2 * args.z_where_size)})

        nn.init.constant_(self.encoder.z_pres.bias, args.z_pres_init_encoder_bias)  # push initial num time steps probs higher

        # initialize STN to identity
        self.encoder.z_where.weight.data.zero_()
        self.encoder.z_where.bias.data = torch.cat([torch.zeros(args.z_where_size), -1.*torch.ones(args.z_where_size)],dim=0)

        # --------------------
        # Baseline model for NVIL per Mnih & Gregor
        # --------------------

        self.baseline = nn.ModuleDict({
            'rnn':       nn.LSTMCell(x_size + args.z_where_size + args.z_what_size + args.z_pres_size, args.baseline_lstm_size),
            'linear':    nn.Linear(args.baseline_lstm_size, 1)})

    @property
    def p_z_pres(self):
        return D.Geometric(probs=1-self.z_pres_prob)

    @property
    def p_z_what(self):
        return D.Normal(self.z_what_mean, self.z_what_scale)

    @property
    def p_z_where(self):
        return D.Normal(self.z_where_mean, self.z_where_scale)

    def forward(self, x, writer=None, box_attn_window_color=None):
        """ cf AIR paper Figure 3 (right) for model flow.
            Computes    (1) inference for z latents;
                        (2) data reconstruction given the latents;
                        (3) baseline for decreasing gradient variance;
                        (4) losses
            Returns
                recon_x       -- tensor of shape (B, C, H, W); reconstruction of data
                pred_counts   -- teonsor of shape (B,); predicted number of object for each data point
                elbo          -- tensor of shape (B,); variational lower bound
                loss          -- tensor of shape (0) of the scalar objective loss
                baseline loss -- tensor of shape (0) of the scalar baseline loss (cf Mnih & Gregor NVIL)
        """
        batch_size = x.shape[0]
        device = x.device

        # store for elbo computation
        pred_counts = torch.zeros(batch_size, self.max_steps, device=device)  # store for object count accuracy
        obj_probs = torch.ones(batch_size, self.max_steps, device=device)     # store for computing the geometric posterior
        baseline = torch.zeros(batch_size, device=device)
        kl_z_pres = torch.zeros(batch_size, device=device)
        kl_z_what = torch.zeros(batch_size, device=device)
        kl_z_where = torch.zeros(batch_size, device=device)

        # initialize canvas, encoder rnn, states of the latent variables, mask for z_pres, baseline rnn
        recon_x = torch.zeros(batch_size, 3 if box_attn_window_color is not None else self.C, self.H, self.W, device=device)
        h_enc = torch.zeros(batch_size, self.lstm_size, device=device)
        c_enc = torch.zeros_like(h_enc)
        z_pres = torch.ones(batch_size, 1, device=device)
        z_what = torch.zeros(batch_size, self.z_what_size, device=device)
        z_where = torch.rand(batch_size, self.z_where_size, device=device)
        h_baseline = torch.zeros(batch_size, self.baseline_lstm_size, device=device)
        c_baseline = torch.zeros_like(h_baseline)

        # run model forward up to a max number of reconstruction steps
        for i in range(self.max_steps):

            # --------------------
            # Inference step -- AIR paper fig3 middle.
            # 1. compute 1-dimensional Bernoulli variable indicating the entity’s presence
            # 2. compute 3-dimensional vector specifying the affine parameters of its position and scale (ziwhere).
            # 3. compute C-dimensional distributed vector describing its class or appearance (ziwhat)
            # --------------------

            # rnn encoder
            h_enc, c_enc = self.encoder.rnn(torch.cat([x, z_pres, z_what, z_where], dim=-1), (h_enc, c_enc))

            # 1. compute 1-dimensional Bernoulli variable indicating the entity’s presence; note: if z_pres == 0, subsequent mask are zeroed
            q_z_pres = D.Bernoulli(probs = torch.clamp(z_pres * torch.sigmoid(self.encoder.z_pres(h_enc)), 1e-5, 1 - 1e-5))  # avoid probs that are exactly 0 or 1
            z_pres = q_z_pres.sample()

            # 2. compute 3-dimensional vector specifying the affine parameters of its position and scale (ziwhere).
            q_z_where_mean, q_z_where_scale = self.encoder.z_where(h_enc).chunk(2, -1)
            q_z_where = D.Normal(q_z_where_mean + self.z_where_mean, F.softplus(q_z_where_scale) * self.z_where_scale)
            z_where = q_z_where.rsample()

            # attend to a part of the image (using a spatial transformer) to produce x_i_att
            x_att = stn(x.view(batch_size, self.C, self.H, self.W), z_where, (batch_size, self.C, self.A, self.A), inverse=False)

            # 3. compute C-dimensional distributed vector describing its class or appearance (ziwhat)
            q_z_what_mean, q_z_what_scale = self.encoder.z_what(x_att.flatten(start_dim=1)).chunk(2, -1)
            q_z_what = D.Normal(q_z_what_mean, F.softplus(q_z_what_scale))
            z_what = q_z_what.rsample()

            # --------------------
            # Reconstruction step
            # 1. computes y_i_att reconstruction of the attention window x_att
            # 2. add to canvas over all timesteps
            # --------------------

            # 1. compute reconstruction of the attention window
            y_att = torch.sigmoid(self.decoder(z_what).view(-1, self.C, self.A, self.A) + self.decoder_bias)

            # scale and shift y according to z_where
            y = stn(y_att, z_where, (batch_size, self.C, self.H, self.W), inverse=True, box_attn_window_color=i if box_attn_window_color is not None else None)

            # 2. add reconstruction to canvas
            recon_x += y * z_pres.view(-1,1,1,1)

            # --------------------
            # Baseline step -- AIR paper cf's Mnih & Gregor NVIL; specifically sec 2.3 variance reduction
            # --------------------

            # compute baseline; independent of the z latents (cf Mnih & Gregor NVIL) so detach from graph
            baseline_input = torch.cat([x, z_pres.detach(), z_what.detach(), z_where.detach()], dim=-1)
            h_baseline, c_baseline = self.baseline.rnn(baseline_input, (h_baseline, c_baseline))
            baseline += self.baseline.linear(h_baseline).squeeze()  # note: masking by z_pres give poorer results

            # --------------------
            # Variational lower bound / loss components
            # --------------------

            # compute kl(q||p) divergences -- sum over latent dim
            kl_z_what += D.kl.kl_divergence(q_z_what, self.p_z_what).sum(1) * z_pres.squeeze()
            kl_z_where += D.kl.kl_divergence(q_z_where, self.p_z_where).sum(1) * z_pres.squeeze()

            pred_counts[:,i] = z_pres.flatten()
            obj_probs[:,i] = q_z_pres.probs.flatten()

        q_z_pres = compute_geometric_from_bernoulli(obj_probs)
        score_fn = q_z_pres[torch.arange(batch_size), pred_counts.sum(1).long()].log()  # log prob of num objects under the geometric
        kl_z_pres = compute_z_pres_kl(q_z_pres, self.p_z_pres, writer).sum(1)  # note: mask by pred_counts makes no difference

        p_x_z = D.Normal(recon_x.flatten(1), self.likelihood_sigma)
        log_like = p_x_z.log_prob(x.view(-1, self.C, self.H, self.W).expand_as(recon_x).flatten(1)).sum(-1) # sum image dims (C, H, W)

        # --------------------
        # Compute variational bound and loss function
        # --------------------

        elbo = log_like - kl_z_pres - kl_z_what - kl_z_where              # objective for loss function, but high variance
        loss = - torch.sum(elbo + (elbo - baseline).detach() * score_fn)  # var reduction surrogate objective objective (cf Mnih & Gregor NVIL)
        baseline_loss = F.mse_loss(elbo.detach(), baseline)

        if writer:
            writer.add_scalar('log_like',       log_like.mean(0).item(), writer.step)
            writer.add_scalar('kl_z_pres',     kl_z_pres.mean(0).item(), writer.step)
            writer.add_scalar('kl_z_what',     kl_z_what.mean(0).item(), writer.step)
            writer.add_scalar('kl_z_where',   kl_z_where.mean(0).item(), writer.step)
            writer.add_scalar('elbo',               elbo.mean(0).item(), writer.step)
            writer.add_scalar('baseline',       baseline.mean(0).item(), writer.step)
            writer.add_scalar('score_function', score_fn.mean(0).item(), writer.step)
            writer.add_scalar('z_pres_prob',    self.z_pres_prob.item(), writer.step)

        return recon_x, pred_counts, elbo, loss, baseline_loss

    @torch.no_grad()
    def generate(self, n_samples):
        """ AIR paper figure 3 left:

        The generative model draws n ∼ Geom(ρ) digits {y_i_att} of size 28 × 28 (two shown), scales andshifts them
        according to z_i_where ∼ N (0, Σ) using spatial transformers, and sums the results {y_i} to form a 50 × 50 image.
        Each digit is obtained by first sampling a latent code z_i_what from the prior z_i_what ∼ N (0, 1) and 
        propagating it through the decoder network of a variational autoencoder.
        The learnable parameters θ of the generative model are the parameters of this decoder network.
        """
        # sample z_pres ~ Geom(rho) -- this is the number of digits present in an image
        z_pres = D.Geometric(1 - self.z_pres_prior).sample((n_samples,)).clamp_(0, self.max_steps)

        # compute a mask on z_pres as e.g.:
        #   z_pres = [1,4,2,0]
        #   mask = [[1,0,0,0,0],
        #           [1,1,1,1,0],
        #           [1,1,0,0,0],
        #           [0,0,0,0,0]]
        #   thus network outputs more objects (sample z_what, z_where and decode) where z_pres is 1
        #   and outputs nothing when z_pres is 0
        z_pres_mask = torch.arange(self.max_steps).float().to(z_pres.device).expand(n_samples, self.max_steps) < z_pres.view(-1,1)
        z_pres_mask = z_pres_mask.float().to(z_pres.device)

        # initialize image canvas
        x = torch.zeros(n_samples, self.C, self.H, self.W).to(z_pres.device)

        # generate digits
        for i in range(int(z_pres.max().item())):  # up until the number of objects sampled via z_pres
            # sample priors
            z_what = self.p_z_what.sample((n_samples,))
            z_where = self.p_z_where.sample((n_samples,))

            # propagate through the decoder, scale and shift y_att according to z_where using spatial transformers
            y_att = torch.sigmoid(self.decoder(z_what).view(n_samples, self.C, self.A, self.A) + self.decoder_bias)
            y = stn(y_att, z_where, (n_samples, self.C, self.H, self.W), inverse=True, box_attn_window_color=i)

            # apply mask and sum results towards final image
            x = x + y * z_pres_mask[:,i].view(-1,1,1,1)
        return x


# --------------------
# Train and evaluate
# --------------------

def train_epoch(model, dataloader, model_optimizer, baseline_optimizer, anneal_z_pres_prob, epoch, writer, args):
    model.train()

    with tqdm(total=len(dataloader), desc='epoch {} / {}'.format(epoch+1, args.start_epoch + args.n_epochs)) as pbar:

        for i, (x, y) in enumerate(dataloader):
            writer.step += 1  # update global step

            x = x.view(x.shape[0], -1).to(args.device)

            # run through model and compute loss
            recon_x, pred_counts, elbo, loss, baseline_loss = model(x, writer if i % args.log_interval == 0 else None)  # pass writer at logging intervals

            # anneal z_pres prior
            model.z_pres_prob = anneal_z_pres_prob(model.z_pres_prob, writer.step, args)

            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

            baseline_optimizer.zero_grad()
            baseline_loss.backward()
            baseline_optimizer.step()

            # update tracking
            count_accuracy = torch.eq(pred_counts.sum(1).cpu(), y.sum(1)).float().mean()
            pbar.set_postfix(elbo='{:.3f}'.format(elbo.mean(0).item()), \
                             loss='{:.3f}'.format(loss.item()), \
                             count_acc='{:.2f}'.format(count_accuracy.item()))
            pbar.update()

            if i % args.log_interval == 0:
                writer.add_scalar('loss', loss.item(), writer.step)
                writer.add_scalar('baseline_loss', baseline_loss.item(), writer.step)
                writer.add_scalar('count_accuracy_train', count_accuracy.item(), writer.step)

    if args.verbose == 1:
        print('z_pres prior:', model.p_z_pres.log_prob(torch.arange(args.max_steps + 1.).to(args.device)).exp(), \
              'post:', compute_geometric_from_bernoulli(pred_counts.mean(0).unsqueeze(0)).squeeze(), \
              'ber success:', pred_counts.mean(0))

@torch.no_grad()
def evaluate(model, dataloader, args, n_samples=10):
    model.eval()

    # initialize trackers
    elbo = 0
    pred_counts = []
    true_counts = []

    # evaluate elbo
    for x, y in tqdm(dataloader):
        x = x.view(x.shape[0], -1).to(args.device)
        _, pred_counts_i, elbo_i, _, _ = model(x)
        elbo += elbo_i.sum(0).item()
        pred_counts += [pred_counts_i.cpu()]
        true_counts += [y]
    elbo /= (len(dataloader) * args.batch_size)

    # evaluate count accuracy; test dataset not shuffled to preds and true aligned sequentially
    pred_counts = torch.cat(pred_counts, dim=0)
    true_counts = torch.cat(true_counts, dim=0)
    count_accuracy = torch.eq(pred_counts.sum(1), true_counts.sum(1)).float().mean()

    # visualize reconstruction
    x = x[-n_samples:]  # take last n_sample data points
    recon_x, _, _, _, _ = model(x, box_attn_window_color=True)
    image_recons = torch.cat([x.view(-1, *args.image_dims).expand_as(recon_x), recon_x], dim=0)
    image_recons = make_grid(image_recons.cpu(), nrow=n_samples, pad_value=1)

    return elbo, count_accuracy, image_recons

def train_and_evaluate(model, train_dataloader, test_dataloader, model_optimizer, baseline_optimizer, anneal_z_pres_prob, writer, args):

    for epoch in range(args.start_epoch, args.start_epoch + args.n_epochs):
        # train
        train_epoch(model, train_dataloader, model_optimizer, baseline_optimizer, anneal_z_pres_prob, epoch, writer, args)

        # evaluate
        if epoch % args.eval_interval == 0:
            test_elbo, count_accuracy, image_recons = evaluate(model, test_dataloader, args)
            print('Evaluation at epoch {}: test elbo {:.3f}; count accuracy {:.3f}'.format(epoch, test_elbo, count_accuracy))
            writer.add_scalar('test_elbo', test_elbo, epoch)
            writer.add_scalar('count_accuracy_test', count_accuracy, epoch)
            writer.add_image('image_reconstruction', image_recons, epoch)
            save_image(image_recons, os.path.join(args.output_dir, 'image_recons_{}.png'.format(epoch)))

            # generate samples
            samples = model.generate(n_samples=10)
            images = make_grid(samples, nrow=samples.shape[0], pad_value=1)
            save_image(images, os.path.join(args.output_dir, 'generated_sample_{}.png'.format(epoch)))
            writer.add_image('training_sample', images, epoch)

            # save training checkpoint
            torch.save({'epoch': epoch,
                        'global_step': writer.step,
                        'state_dict': model.state_dict()},
                        os.path.join(args.output_dir, 'checkpoint.pt'))


# --------------------
# Main
# --------------------

if __name__ == '__main__':
    args = parser.parse_args()

    # setup writer and output folders
    writer = SummaryWriter(log_dir = os.path.join(args.output_dir, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime())) \
                                        if not args.restore_file else os.path.dirname(args.restore_file))
    writer.step = 0
    args.output_dir = writer.file_writer.get_logdir()  # update output_dir with the writer unique directory

    # setup device
    args.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda is not None else 'cpu')
    torch.manual_seed(args.seed)
    if args.device.type == 'cuda': torch.cuda.manual_seed(args.seed)

    # load data
    train_dataloader, test_dataloader = fetch_dataloaders(args)

    # load model
    model = AIR(args).to(args.device)

    # load optimizers
    model_optimizer = torch.optim.RMSprop(model.parameters(), lr=args.model_lr, momentum=0.9)
    baseline_optimizer = torch.optim.RMSprop(model.parameters(), lr=args.baseline_lr, momentum=0.9)

    if args.restore_file:
        checkpoint = torch.load(args.restore_file, map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'])
        writer.step = checkpoint['global_step']
        args.start_epoch = checkpoint['epoch'] + 1
        # set up paths
        args.output_dir = os.path.dirname(args.restore_file)

    # save settings
    with open(os.path.join(args.output_dir, 'config.txt'), 'a') as f:
        print('Parsed args:\n', pprint.pformat(args.__dict__), file=f)
        print('\nModel:\n', model, file=f)

    if args.train:
        train_and_evaluate(model, train_dataloader, test_dataloader, model_optimizer, baseline_optimizer, anneal_z_pres_prob, writer, args)

    if args.evaluate:
        test_elbo, count_accuracy, image_recons = evaluate(model, test_dataloader, args)
        print('Evaluation: test elbo {:.3f}; {:.3f}'.format(test_elbo, count_accuracy))
        save_image(image_recons, os.path.join(args.output_dir, 'image_recons.png'))

    if args.generate:
        samples = model.generate(n_samples=7)
        images = make_grid(samples, pad_value=1)
        save_image(images, os.path.join(args.output_dir, 'generated_sample.png'))
        writer.add_image('generated_sample', images)

    writer.close()
