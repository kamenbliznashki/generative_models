"""
DRAW: A Recurrent Neural Network For Image Generation
https://arxiv.org/pdf/1502.04623.pdf
"""

import os
import argparse
import time
from tqdm import tqdm
import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid

import utils

parser = argparse.ArgumentParser()

# model parameters
parser.add_argument('--image_dims', type=tuple, default=(1,28,28), help='Dimensions of a single datapoint (e.g. (1,28,28) for MNIST).')
parser.add_argument('--time_steps', type=int, default=32, help='Number of time-steps T consumed by the network before performing reconstruction.')
parser.add_argument('--z_size', type=int, default=100, help='Size of the latent representation.')
parser.add_argument('--lstm_size', type=int, default=256, help='Size of the hidden layer in the encoder/decoder models.')
parser.add_argument('--read_size', type=int, default=2, help='Size of the read operation visual field.')
parser.add_argument('--write_size', type=int, default=5, help='Size of the write operation visual field.')
parser.add_argument('--use_read_attn', action='store_true', help='Whether to use visual attention or not. If not, read/write field size is the full image.')
parser.add_argument('--use_write_attn', action='store_true', help='Whether to use visual attention or not. If not, read/write field size is the full image.')

# training params
parser.add_argument('--train', action='store_true')
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--log_interval', default=100, help='How often to write summary outputs.')
parser.add_argument('--cuda', type=int, help='Which cuda device to use')
parser.add_argument('--mini_data', action='store_true')
parser.add_argument('--verbose', '-v', action='store_true', help='Extra monitoring of training + record forward/backward hooks on attn params')

# eval params
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--test_batch_size', type=int, default=10, help='Batch size for evaluation')

# generate params
parser.add_argument('--generate', action='store_true')

# data paths
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--data_dir', default='./data')
parser.add_argument('--output_dir', default='./results/{}'.format(os.path.splitext(__file__)[0]))
parser.add_argument('--restore_file', help='Path to .pt checkpoint file.')




# --------------------
# Data
# --------------------

def fetch_dataloader(args, batch_size, train=True, download=False, mini_size=128):

    transforms = T.Compose([T.ToTensor()])
    dataset = MNIST(root=args.data_dir, train=train, download=download, transform=transforms)

    # load dataset and init in the dataloader
    if args.mini_data:
        if train:
            dataset.train_data = dataset.train_data[:mini_size]
            dataset.train_labels = dataset.train_labels[:mini_size]
        else:
            dataset.test_data = dataset.test_data[:mini_size]
            dataset.test_labels = dataset.test_labels[:mini_size]


    kwargs = {'num_workers': 1, 'pin_memory': True} if args.device.type is 'cuda' else {}

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=True, **kwargs)


# --------------------
# Model
# --------------------

class DRAW(nn.Module):
    def __init__(self, args):
        super().__init__()

        # save dimensions
        self.C, self.H, self.W = args.image_dims
        self.time_steps = args.time_steps
        self.lstm_size = args.lstm_size
        self.z_size = args.z_size
        self.use_read_attn = args.use_read_attn
        self.use_write_attn = args.use_write_attn
        if self.use_read_attn:
            self.read_size = args.read_size
        else:
            self.read_size = self.H
        if self.use_write_attn:
            self.write_size = args.write_size
        else:
            self.write_size = self.H

        # encoder - decoder layers
        self.encoder = nn.LSTMCell(2 * self.read_size * self.read_size + self.lstm_size, self.lstm_size)
        self.decoder = nn.LSTMCell(self.z_size, self.lstm_size)

        # latent space layer
        # outputs the parameters of the q distribution (mu and var; here q it is a Normal)
        self.z_linear = nn.Linear(self.lstm_size, 2 * self.z_size)

        # write layers
        self.write_linear = nn.Linear(self.lstm_size, self.write_size * self.write_size)

        # filter bank
        # outputs center location (g_x, g_y), stride delta, logvar of the gaussian filters, scalar intensity gamma (5 params in total)
        self.read_attention_linear = nn.Linear(self.lstm_size, 5)
        self.write_attention_linear = nn.Linear(self.lstm_size, 5)

    def compute_q_distribution(self, h_enc):
        mus, logsigmas = torch.split(self.z_linear(h_enc), [self.z_size, self.z_size], dim=1)
        sigmas = logsigmas.exp()
        z_dist = D.Normal(loc=mus, scale=sigmas)
        return z_dist.rsample(), mus, sigmas

    def read(self, x, x_hat, h_dec):
        if self.use_read_attn:
            # read filter bank -- eq 21
            g_x, g_y, logvar, logdelta, loggamma = self.read_attention_linear(h_dec).split(split_size=1, dim=1)  # split returns column vecs
            # compute filter bank matrices -- eq 22 - 26
            g_x, g_y, delta, mu_x, mu_y, F_x, F_y = compute_filterbank_matrices(g_x, g_y, logvar, logdelta, self.H, self.W, self.read_size)
            # output reading wth attention -- eq 27
            new_x     = F_y @ x.view(-1, self.H, self.W)     @ F_x.transpose(-2, -1)   # out (B, N, N)
            new_x_hat = F_y @ x_hat.view(-1, self.H, self.W) @ F_x.transpose(-2, -1)   # out (B, N, N)
            return loggamma.exp() * torch.cat([new_x.view(x.shape[0], -1), new_x_hat.view(x.shape[0], -1)], dim=1)
        else:
            # output reading without attention -- eq 17
            return torch.cat([x, x_hat], dim=1)

    def write(self, h_dec):
        w = self.write_linear(h_dec)

        if self.use_write_attn:
            # read filter bank -- eq 21
            g_x, g_y, logvar, logdelta, loggamma = self.write_attention_linear(h_dec).split(split_size=1, dim=1)  # split returns column vecs
            # compute filter bank matrices -- eq 22 - 26
            g_x, g_y, delta, mu_x, mu_y, F_x, F_y = compute_filterbank_matrices(g_x, g_y, logvar, logdelta, self.H, self.W, self.write_size)
            # output write with attention -- eq 29
            w = F_y.transpose(-2, -1) @ w.view(-1, self.write_size, self.write_size) @ F_x
            return 1. / loggamma.exp() * w.view(w.shape[0], -1)
        else:
            # output write without attention -- eq 18
            return w

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device

        # record metrics for loss calculation
        mus, sigmas = [0]*self.time_steps, [0]*self.time_steps

        # initialize the canvas matrix c on same device and the hidden state and cell state for the encoder and decoder
        c = torch.zeros(*x.shape).to(device)
        h_enc = torch.zeros(batch_size, self.lstm_size).to(device)
        c_enc = torch.zeros_like(h_enc)
        h_dec = torch.zeros(batch_size, self.lstm_size).to(device)
        c_dec = torch.zeros_like(h_dec)

        # run model forward (cf DRAW eq 3 - 8)
        for t in range(self.time_steps):
            x_hat = x.to(c.device) - torch.sigmoid(c)
            r = self.read(x, x_hat, h_dec)
            h_enc, c_enc = self.encoder(torch.cat([r, h_dec], dim=1), (h_enc, c_enc))
            z_sample, mus[t], sigmas[t] = self.compute_q_distribution(h_enc)
            h_dec, c_dec = self.decoder(z_sample, (h_dec, c_dec))
            c = c + self.write(h_dec)

        # return
        #   data likelihood; used to compute L_x loss -- shape (B, H*W)
        #   sequence of latent distributions Q; used to compute L_z loss (here Normal of shape (B, z_size, time_steps)
        return D.Bernoulli(logits=c), D.Normal(torch.stack(mus, dim=-1), torch.stack(sigmas, dim=-1))

    @torch.no_grad()
    def generate(self, n_samples, args):
        samples_time_seq = []

        # initialize model
        c = torch.zeros(n_samples, self.C * self.H * self.W).to(args.device)
        h_dec = torch.zeros(n_samples, self.lstm_size).to(args.device)
        c_dec = torch.zeros_like(h_dec).to(args.device)

        # run for the number of time steps
        for t in range(self.time_steps):
            z_sample = D.Normal(0,1).sample((n_samples, self.z_size)).to(args.device)
            h_dec, c_dec = self.decoder(z_sample, (h_dec, c_dec))
            c = c + self.write(h_dec)
            x = D.Bernoulli(logits=c.view(n_samples, self.C, self.H, self.W)).probs

            samples_time_seq.append(x)

        return samples_time_seq


def compute_filterbank_matrices(g_x, g_y, logvar, logdelta, H, W, attn_window_size):
    """ DRAW section 3.2 -- computes the parameters for an NxN grid of Gaussian filters over the input image.
    Args
        g_x, g_y -- tensors of shape (B, 1); unnormalized center coords for the attention window
        logvar -- tensor of shape (B, 1); log variance for the Gaussian filters (filterbank matrices) on the attention window
        logdelta -- tensor of shape (B, 1); unnormalized stride for the spacing of the filters in the attention window
        H, W -- scalars; original image dimensions
        attn_window_size -- scalar; size of the attention window (specified by the read_size / write_size input args

    Returns
        g_x, g_y -- tensors of shape (B, 1); normalized center coords of the attention window;
        delta -- tensor of shape (B, 1); stride for the spacing of the filters in the attention window
        mu_x, mu_y -- tensors of shape (B, attn_window_size); means location of the filters at row and column
        F_x, F_y -- tensors of shape (B, N, W) and (B, N, H) where N=attention_window_size; filterbank matrices
    """

    batch_size = g_x.shape[0]
    device = g_x.device

    # rescale attention window center coords and stride to ensure the initial patch covers the whole input image
    # eq 22 - 24
    g_x = 0.5 * (W + 1) * (g_x + 1)  # (B, 1)
    g_y = 0.5 * (H + 1) * (g_y + 1)  # (B, 1)
    delta = (max(H, W) - 1) / (attn_window_size - 1) * logdelta.exp()  # (B, 1)

    # compute the means of the filter
    # eq 19 - 20
    mu_x = g_x + (torch.arange(1., 1. + attn_window_size).to(device) - 0.5*(attn_window_size + 1)) * delta  # (B, N)
    mu_y = g_y + (torch.arange(1., 1. + attn_window_size).to(device) - 0.5*(attn_window_size + 1)) * delta  # (B, N)

    # compute the filterbank matrices
    # B = batch dim; N = attn window size; H = original heigh; W = original width
    # eq 25 -- combines logvar=(B, 1, 1) * ( range=(B, 1, W) - mu=(B, N, 1) ) = out (B, N, W); then normalizes over W dimension;
    F_x = torch.exp(- 0.5 / logvar.exp().view(-1,1,1) * (torch.arange(1., 1. + W).repeat(batch_size, 1, 1).to(device) - mu_x.unsqueeze(-1))**2)
    F_x = F_x / torch.sum(F_x + 1e-8, dim=2, keepdim=True)  # normalize over the coordinates of the input image
    # eq 26
    F_y = torch.exp(- 0.5 / logvar.exp().view(-1,1,1) * (torch.arange(1., 1. + H).repeat(batch_size, 1, 1).to(device) - mu_y.unsqueeze(-1))**2)
    F_y = F_y / torch.sum(F_y + 1e-8, dim=2, keepdim=True)  # normalize over the coordinates of the input image

    # returns DRAW paper eq 22, 23, 24, 19, 20, 25, 26
    return g_x, g_y, delta, mu_x, mu_y, F_x, F_y


def loss_fn(d, q, x, writer=None, step=None):
    """
    Args
        d -- data likelihood distribution output by the model (Bernoulli)
        q -- approximation distribution to the latent variable z (Normal)
    """
    # cf DRAW paper section 2

    # reconstruction loss L_x eq 9 -- negative log probability of x under the model d
    #                                 (sum log probs over the pixels of each datapoint and mean over batch dim)
    # latent loss L_z eq 10 -- sum KL over temporal dimension (number of time steps) and mean over batch and z dims
    batch_size = x.shape[0]
    p_prior = D.Normal(torch.tensor(0., device=x.device), torch.tensor(1., device=x.device))
    loss_log_likelihood = - d.log_prob(x).sum(-1).mean(0)                # sum over pixels (-1), mean over datapoints (0)
    loss_kl = D.kl.kl_divergence(q, p_prior).sum(dim=[-2,-1]).mean(0)  # sum over time_steps (-1) and z (-2), mean over datapoints (0)

    if writer:
        writer.add_scalar('loss_log_likelihood', loss_log_likelihood, step)
        writer.add_scalar('loss_kl', loss_kl, step)

    return loss_log_likelihood + loss_kl


# --------------------
# Train and eval
# --------------------

def train_epoch(model, dataloader, loss_fn, optimizer, epoch, writer, args):
    model.train()

    with tqdm(total=len(dataloader), desc='epoch {} of {}'.format(epoch+1, args.n_epochs)) as pbar:
        time.sleep(0.1)

        for i, (x, _) in enumerate(dataloader):
            global_step = epoch * len(dataloader) + i + 1

            x = x.view(x.shape[0], -1).to(args.device)

            (d, q) = model(x)

            loss = loss_fn(d, q, x, writer, global_step)

            optimizer.zero_grad()
            loss.backward()

            # record grad norm and clip
            if args.verbose:
                grad_norm = 0
                for name, p in model.named_parameters():
                    grad_norm += p.grad.norm().item() if p.grad is not None else 0
                writer.add_scalar('grad_norm', grad_norm, global_step)
            nn.utils.clip_grad_norm_(model.parameters(), 10)

            optimizer.step()

            # update tracking
            pbar.set_postfix(loss='{:.3f}'.format(loss.item()))
            pbar.update()

            if i % args.log_interval == 0:
                writer.add_scalar('loss', loss.item(), global_step)


def train_and_evaluate(model, train_dataloader, test_dataloader, loss_fn, optimizer, writer, args):
    start_epoch = 0

    if args.restore_file:
        print('Restoring parameters from {}'.format(args.restore_file))
        start_epoch = utils.load_checkpoint(args.restore_file, [model], [optimizer], map_location=args.device.type)
        args.n_epochs += start_epoch - 1
        print('Resuming training from epoch {}'.format(start_epoch))

    for epoch in range(start_epoch, args.n_epochs):
        train_epoch(model, train_dataloader, loss_fn, optimizer, epoch, writer, args)
#        evaluate(model, test_dataloader, loss_fn, writer, args, epoch)

        # snapshot at end of epoch
        if args.save_model:
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'model_state_dicts': [model.state_dict()],
                                   'optimizer_state_dicts': [optimizer.state_dict()]},
                                   checkpoint=args.output_dir,
                                   quiet=True)


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, writer, args, epoch=None):
    model.eval()

    # sample the generation model
    samples_time_seq = model.generate(args.test_batch_size**2, args)
    samples = samples_time_seq[-1] # grab the final sample
    # pull targets to search closest neighbor to
    #   right-most column of a nxn image grid where n is args.test_batch_size -- start at index n-1 and skip by n (e.g 9, 19, ...)
    targets = samples[args.test_batch_size - 1 :: args.test_batch_size].view(args.test_batch_size, -1)
    # initialize a large max L2 pixel distance and tensor for l2-closest neighbors
    max_distances = 100 * samples[0].numel() * torch.ones(args.test_batch_size).to(args.device)
    closest_neighbors = torch.zeros_like(targets)

    # compute ELBO on dataset
    cum_loss = 0
    for x, _ in tqdm(dataloader):
        x = x.view(x.shape[0], -1).float().to(args.device)

        # run through model and aggregate loss
        d, q = model(x)
        # aggregate loss
        loss = loss_fn(d, q, x)
        cum_loss += loss.item()

        # find closest neighbors to the targets sampled above - l2 distance between targets and images in minibatch
        distances = F.pairwise_distance(x, targets)
        mask = distances < max_distances
        max_distances[mask] = distances[mask]
        closest_neighbors[mask] = x[mask]

    cum_loss /= len(dataloader)
    # output loss
    print('Evaluation ELBO: {:.2f}'.format(cum_loss))
    writer.add_scalar('Evaluation ELBO', cum_loss, epoch)

    # visualize generated samples and closest neighbors (cf DRAW paper fig 6)
    generated = make_grid(samples.cpu(), nrow=samples.shape[0]//args.test_batch_size)
    spacer = torch.ones_like(generated)[:,:,:2]
    neighbors = make_grid(closest_neighbors.view(-1, *args.image_dims).cpu(), nrow=1)
    images = torch.cat([generated, spacer, neighbors], dim=-1)
    save_image(images, os.path.join(args.output_dir,
                                    'evaluation_sample' + (epoch!=None)*'_epoch_{}'.format(epoch) + '.png'))
    writer.add_image('generated images', images, epoch)


@torch.no_grad()
def generate(model, writer, args, n_samples=64):
    import math

    # generate samples
    samples_time_seq = model.generate(n_samples, args)

    # visualize generation sequence (cf DRAW paper fig 7)
    images = torch.stack(samples_time_seq, dim=1).view(-1, *args.image_dims)  # reshape to (10*time_steps, 1, 28, 28)
    images = make_grid(images, nrow=len(samples_time_seq), padding=1, pad_value=1)
    save_name = 'generated_sequences_r{}_w{}_steps{}.png'.format(args.read_size, args.write_size, args.time_steps)
    save_image(images, os.path.join(args.output_dir, save_name))
    writer.add_image(save_name, images)

    # make gif
    for i in range(len(samples_time_seq)):
        # convert sequence of image tensors to 8x8 grid
        image = make_grid(samples_time_seq[i].cpu(), nrow=int(math.sqrt(n_samples)), padding=1, normalize=True, pad_value=1)
        # make into gif
        samples_time_seq[i] = image.data.numpy().transpose(1,2,0)

    import imageio
    imageio.mimsave(os.path.join(args.output_dir, 'generated_{}_time_steps.gif'.format(args.time_steps)), samples_time_seq)


# --------------------
# Monitor training
# --------------------

def record_attn_params(self, in_tensor, out_tensor, bank_name):
    g_x, g_y, logvar, logdelta, loggamma = out_tensor.cpu().split(split_size=1, dim=1)
    writer.add_scalar(bank_name + ' g_x', g_x.mean())
    writer.add_scalar(bank_name + ' g_y', g_y.mean())
    writer.add_scalar(bank_name + ' var', logvar.exp().mean())
    writer.add_scalar(bank_name + ' exp_logdelta', logdelta.exp().mean())
    writer.add_scalar(bank_name + ' gamma', loggamma.exp().mean())

def record_attn_grads(self, in_tensor, out_tensor, bank_name):
    g_x, g_y, logvar, logdelta, loggamma = out_tensor[0].cpu().split(split_size=1, dim=1)
    writer.add_scalar(bank_name + ' grad_var', logvar.exp().mean())
    writer.add_scalar(bank_name + ' grad_logdelta', logdelta.exp().mean())

def record_forward_backward_attn_hooks(model):
    from functools import partial

    model.read_attention_linear.register_forward_hook(partial(record_attn_params, bank_name='read'))
    model.write_attention_linear.register_forward_hook(partial(record_attn_params, bank_name='write'))
    model.write_attention_linear.register_backward_hook(partial(record_attn_grads, bank_name='write'))


# --------------------
# Main
# --------------------

if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    writer = utils.set_writer(args.output_dir if args.restore_file is None else args.restore_file,
                              (args.restore_file==None)*'',  # suffix only when not restoring
                              args.restore_file is not None)
    # update output_dir with the writer unique directory
    args.output_dir = writer.file_writer.get_logdir()

    args.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda is not None else 'cpu')

    # set seed
    torch.manual_seed(5)
    if args.device.type is 'cuda': torch.cuda.manual_seed(11192018)

    # set up model
    model = DRAW(args).to(args.device)

    if args.verbose:
        record_forward_backward_attn_hooks(model)

    # train
    if args.train:
        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        # dataloaders
        train_dataloader = fetch_dataloader(args, args.train_batch_size, train=True)
        test_dataloader = fetch_dataloader(args, args.test_batch_size, train=False)
        # run training
        print('Starting training with args:\n', args)
        writer.add_text('Params', pprint.pformat(args.__dict__))
        with open(os.path.join(args.output_dir, 'params.txt'), 'w') as f:
            pprint.pprint(args.__dict__, f)
        train_and_evaluate(model, train_dataloader, test_dataloader, loss_fn, optimizer, writer, args)

    # eval
    if args.evaluate:
        print('Restoring parameters from {}'.format(args.restore_file))
        _ = utils.load_checkpoint(args.restore_file, [model])
        print('Evaluating model with args:\n', args)
        # get test dataloader
        dataloader = fetch_dataloader(args, args.test_batch_size, train=False)
        # evaluate
        evaluate(model, dataloader, loss_fn, writer, args)

    # generate
    if args.generate:
        print('Restoring parameters from {}'.format(args.restore_file))
        _ = utils.load_checkpoint(args.restore_file, [model])
        print('Generating images from model with args:\n', args)
        generate(model, writer, args)


    writer.close()

