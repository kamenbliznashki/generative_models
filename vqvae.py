"""
Implementation of VQ-VAE-2:
    -- van den Oord, 'Generating Diverse High-Fidelity Images with VQ-VAE-2' -- https://arxiv.org/abs/1906.00446
    -- van den Oord, 'Neural Discrete Representation Learning' -- https://arxiv.org/abs/1711.00937
    -- Roy, Theory and Experiments on Vector Quantized Autoencoders' -- https://arxiv.org/pdf/1805.11063.pdf

Reference implementation of the vector quantized VAE:
    https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image, make_grid

from tensorboardX import SummaryWriter
from tqdm import tqdm

import os
import argparse
import time
import json
import pprint
from functools import partial

from datasets.chexpert import ChexpertDataset

parser = argparse.ArgumentParser()
# action
parser.add_argument('--train', action='store_true', help='Train model.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate model.')
parser.add_argument('--generate', action='store_true', help='Generate samples from a model.')
parser.add_argument('--seed', type=int, default=0, help='Random seed to use.')
parser.add_argument('--cuda', type=int, help='Which cuda device to use.')
parser.add_argument('--mini_data', action='store_true', help='Truncate dataset to a single minibatch.')
# model
parser.add_argument('--n_embeddings', default=256, type=int, help='Size of discrete latent space (K-way categorical).')
parser.add_argument('--embedding_dim', default=64, type=int, help='Dimensionality of each latent embedding vector.')
parser.add_argument('--n_channels', default=128, type=int, help='Number of channels in the encoder and decoder.')
parser.add_argument('--n_res_channels', default=64, type=int, help='Number of channels in the residual layers.')
parser.add_argument('--n_res_layers', default=2, type=int, help='Number of residual layers inside the residual block.')
parser.add_argument('--n_cond_classes', type=int, help='(NOT USED here; used in training prior but requires flag for dataloader) Number of classes if conditional model.')
# data params
parser.add_argument('--dataset', choices=['cifar10', 'chexpert'], default='chexpert')
parser.add_argument('--data_dir', default='~/data/', help='Location of datasets.')
parser.add_argument('--output_dir', type=str, help='Location where weights, logs, and sample should be saved.')
parser.add_argument('--restore_dir', type=str, help='Path to model config and checkpoint to restore.')
# training param
parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
parser.add_argument('--lr_decay', type=float, default=0.9999965, help='Learning rate decay (assume end lr = 1e-6 @ 2m iters for init lr 0.001).')
parser.add_argument('--commitment_cost', type=float, default=0.25, help='Commitment cost term in loss function.')
parser.add_argument('--ema', action='store_true', help='Use exponential moving average training for the codebook.')
parser.add_argument('--ema_decay', type=float, default=0.99, help='EMA decay rate.')
parser.add_argument('--ema_eps', type=float, default=1e-5, help='EMA epsilon.')
parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--step', type=int, default=0, help='Current step of training (number of minibatches processed).')
parser.add_argument('--start_epoch', default=0, help='Starting epoch (for logging; to be overwritten when restoring file.')
parser.add_argument('--log_interval', type=int, default=50, help='How often to show loss statistics and save samples.')
parser.add_argument('--eval_interval', type=int, default=10, help='How often to evaluate and save samples.')
# distributed training params
parser.add_argument('--distributed', action='store_true', default=False, help='Whether to use DistributedDataParallels on multiple machines and GPUs.')
# generation param
parser.add_argument('--n_samples', type=int, default=64, help='Number of samples to generate.')


# --------------------
# Data and model loading
# --------------------

def fetch_vqvae_dataloader(args, train=True):
    if args.dataset == 'cifar10':
        # setup dataset and dataloader -- preprocess data to [-1, 1]
        dataset = CIFAR10(args.data_dir,
                          train=train,
                          transform=T.Compose([T.ToTensor(), lambda x: x.mul(2).sub(1)]),
                          target_transform=(lambda y: torch.eye(args.n_cond_classes)[y]) if args.n_cond_classes else None)
        if not 'input_dims' in args: args.input_dims = (3,32,32)
    elif args.dataset == 'chexpert':
        dataset = ChexpertDataset(args.data_dir, train,
                                  transform=T.Compose([T.ToTensor(), lambda x: x.mul(2).sub(1)]))
        if not 'input_dims' in args: args.input_dims = dataset.input_dims
        args.n_cond_classes = len(dataset.attr_idxs)

    if args.mini_data:
        dataset.data = dataset.data[:args.batch_size]
    return DataLoader(dataset, args.batch_size, shuffle=train, num_workers=4, pin_memory=('cuda' in args.device))

def load_model(model_cls, config, model_dir, args, restore=False, eval_mode=False, optimizer_cls=None, scheduler_cls=None, verbose=True):
    # load model config
    if config is None: config = load_json(os.path.join(model_dir, 'config_{}.json'.format(args.cuda)))
    # init model and distribute
    model = model_cls(**config).to(args.device)
    if args.distributed:
        # NOTE: DistributedDataParallel will divide and allocate batch_size to all available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.cuda], output_device=args.cuda,
                                                          find_unused_parameters=True)
    # init optimizer and scheduler
    optimizer = optimizer_cls(model.parameters()) if optimizer_cls else None
    scheduler = scheduler_cls(optimizer) if scheduler_cls else None
    if restore:
        checkpoint = torch.load(os.path.join(model_dir, 'checkpoint.pt'), map_location=args.device)
        if args.distributed:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch'] + 1
        args.step = checkpoint['global_step']
        if optimizer: optimizer.load_state_dict(torch.load(model_dir + '/optim_checkpoint.pt', map_location=args.device))
        if scheduler: scheduler.load_state_dict(torch.load(model_dir + '/sched_checkpoint.pt', map_location=args.device))
    if eval_mode:
        model.eval()
#        if optimizer and restore: optimizer.use_ema(True)
        for p in model.parameters(): p.requires_grad_(False)
    if verbose:
        print('Loaded {}\n\tconfig and state dict loaded from {}'.format(model_cls.__name__, model_dir))
        print('\tmodel parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))
    return model, optimizer, scheduler

def save_json(data, filename, args):
    with open(os.path.join(args.output_dir, filename + '.json'), 'w') as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# --------------------
# VQVAE components
# --------------------

class VQ(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, ema=False, ema_decay=0.99, ema_eps=1e-5):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.ema = ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps

        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        nn.init.kaiming_uniform_(self.embedding.weight, 1)

        if ema:
            self.embedding.weight.requires_grad_(False)
            # set up moving averages
            self.register_buffer('ema_cluster_size', torch.zeros(n_embeddings))
            self.register_buffer('ema_weight', self.embedding.weight.clone().detach())

    def embed(self, encoding_indices):
        return self.embedding(encoding_indices).permute(0,4,1,2,3).squeeze(2)  # in (B,1,H,W); out (B,E,H,W)

    def forward(self, z):
        # input (B,E,H,W); permute and reshape to (B*H*W,E) to compute distances in E-space
        flat_z = z.permute(0,2,3,1).reshape(-1, self.embedding_dim)   # (B*H*W,E)
        # compute distances to nearest embedding
        distances = flat_z.pow(2).sum(1, True) + self.embedding.weight.pow(2).sum(1) - 2 * flat_z.matmul(self.embedding.weight.t())
        # quantize z to nearest embedding
        encoding_indices = distances.argmin(1).reshape(z.shape[0], 1, *z.shape[2:])   # (B,1,H,W)
        z_q = self.embed(encoding_indices)

        # perform ema updates
        if self.ema and self.training:
            with torch.no_grad():
                # update cluster size
                encodings = F.one_hot(encoding_indices.flatten(), self.n_embeddings).float().to(z.device)
                self.ema_cluster_size -= (1 - self.ema_decay) * (self.ema_cluster_size - encodings.sum(0))
                # update weight
                dw = z.permute(1,0,2,3).flatten(1) @ encodings  # (E,B*H*W) dot (B*H*W,n_embeddings)
                self.ema_weight -= (1 - self.ema_decay) * (self.ema_weight - dw.t())
                # update embedding weight with normalized ema_weight
                n = self.ema_cluster_size.sum()
                updated_cluster_size = (self.ema_cluster_size + self.ema_eps) / (n + self.n_embeddings * self.ema_eps) * n
                self.embedding.weight.data = self.ema_weight / updated_cluster_size.unsqueeze(1)

        return encoding_indices, z_q   # out (B,1,H,W) codes and (B,E,H,W) embedded codes


class ResidualLayer(nn.Sequential):
    def __init__(self, n_channels, n_res_channels):
        super().__init__(nn.Conv2d(n_channels, n_res_channels, kernel_size=3, padding=1),
                         nn.ReLU(True),
                         nn.Conv2d(n_res_channels, n_channels, kernel_size=1))

    def forward(self, x):
        return F.relu(x + super().forward(x), True)

# --------------------
# VQVAE2
# --------------------

class VQVAE2(nn.Module):
    def __init__(self, input_dims, n_embeddings, embedding_dim, n_channels, n_res_channels, n_res_layers,
                 ema=False, ema_decay=0.99, ema_eps=1e-5, **kwargs):   # keep kwargs so can load from config with arbitrary other args
        super().__init__()
        self.ema = ema

        self.enc1 = nn.Sequential(nn.Conv2d(input_dims[0], n_channels//2, kernel_size=4, stride=2, padding=1),
                                  nn.ReLU(True),
                                  nn.Conv2d(n_channels//2, n_channels, kernel_size=4, stride=2, padding=1),
                                  nn.ReLU(True),
                                  nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
                                  nn.ReLU(True),
                                  nn.Sequential(*[ResidualLayer(n_channels, n_res_channels) for _ in range(n_res_layers)]),
                                  nn.Conv2d(n_channels, embedding_dim, kernel_size=1))

        self.enc2 = nn.Sequential(nn.Conv2d(embedding_dim, n_channels//2, kernel_size=4, stride=2, padding=1),
                                  nn.ReLU(True),
                                  nn.Conv2d(n_channels//2, n_channels, kernel_size=3, padding=1),
                                  nn.ReLU(True),
                                  nn.Sequential(*[ResidualLayer(n_channels, n_res_channels) for _ in range(n_res_layers)]),
                                  nn.Conv2d(n_channels, embedding_dim, kernel_size=1))

        self.dec2 = nn.Sequential(nn.Conv2d(embedding_dim, n_channels, kernel_size=3, padding=1),
                                  nn.ReLU(True),
                                  nn.Sequential(*[ResidualLayer(n_channels, n_res_channels) for _ in range(n_res_layers)]),
                                  nn.ConvTranspose2d(n_channels, embedding_dim, kernel_size=4, stride=2, padding=1))

        self.dec1 = nn.Sequential(nn.Conv2d(2*embedding_dim, n_channels, kernel_size=3, padding=1),
                                  nn.ReLU(True),
                                  nn.Sequential(*[ResidualLayer(n_channels, n_res_channels) for _ in range(n_res_layers)]),
                                  nn.ConvTranspose2d(n_channels, n_channels//2, kernel_size=4, stride=2, padding=1),
                                  nn.ReLU(True),
                                  nn.ConvTranspose2d(n_channels//2, input_dims[0], kernel_size=4, stride=2, padding=1))

        self.proj_to_vq1 = nn.Conv2d(2*embedding_dim, embedding_dim, kernel_size=1)
        self.upsample_to_dec1 = nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1)

        self.vq1 = VQ(n_embeddings, embedding_dim, ema, ema_decay, ema_eps)
        self.vq2 = VQ(n_embeddings, embedding_dim, ema, ema_decay, ema_eps)

    def encode(self, x):
        z1 = self.enc1(x)
        z2 = self.enc2(z1)
        return (z1, z2)  # each is (B,E,H,W)

    def embed(self, encoding_indices):
        encoding_indices1, encoding_indices2 = encoding_indices
        return (self.vq1.embed(encoding_indices1), self.vq2.embed(encoding_indices2))

    def quantize(self, z_e):
        # unpack inputs
        z1, z2 = z_e

        # quantize top level
        encoding_indices2, zq2 = self.vq2(z2)

        # quantize bottom level conditioned on top level decoder and bottom level encoder
        #   decode top level
        quantized2 = z2 + (zq2 - z2).detach()  # stop decoder optimization from accessing the embedding
        dec2_out = self.dec2(quantized2)
        #   condition on bottom encoder and top decoder
        vq1_input = torch.cat([z1, dec2_out], 1)
        vq1_input = self.proj_to_vq1(vq1_input)
        encoding_indices1, zq1 = self.vq1(vq1_input)
        return (encoding_indices1, encoding_indices2), (zq1, zq2)

    def decode(self, z_e, z_q):
        # unpack inputs
        zq1, zq2 = z_q
        if z_e is not None:
            z1, z2 = z_e
            # stop decoder optimization from accessing the embedding
            zq1 = z1 + (zq1 - z1).detach()
            zq2 = z2 + (zq2 - z2).detach()

        # upsample quantized2 to match spacial dim of quantized1
        zq2_upsampled = self.upsample_to_dec1(zq2)
        # decode
        combined_latents = torch.cat([zq1, zq2_upsampled], 1)
        return self.dec1(combined_latents)

    def forward(self, x, commitment_cost, writer=None):
        # Figure 2a in paper
        z_e = self.encode(x)
        encoding_indices, z_q = self.quantize(z_e)
        recon_x = self.decode(z_e, z_q)

        # compute loss over the hierarchy -- cf eq 2 in paper
        recon_loss    = F.mse_loss(recon_x, x)
        q_latent_loss = sum(F.mse_loss(z_i.detach(), zq_i) for z_i, zq_i in zip(z_e, z_q)) if not self.ema else torch.zeros(1, device=x.device)
        e_latent_loss = sum(F.mse_loss(z_i, zq_i.detach()) for z_i, zq_i in zip(z_e, z_q))
        loss = recon_loss + q_latent_loss + commitment_cost * e_latent_loss

        if writer:
            # compute perplexity
            n_embeddings = self.vq1.embedding.num_embeddings
            avg_probs = lambda e: torch.histc(e.float(), bins=n_embeddings, max=n_embeddings).float().div(e.numel())
            perplexity = lambda avg_probs: torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            # record training stats
            writer.add_scalar('loss', loss.item(), args.step)
            writer.add_scalar('loss_recon_train', recon_loss.item(), args.step)
            writer.add_scalar('loss_q_latent', q_latent_loss.item(), args.step)
            writer.add_scalar('loss_e_latent', e_latent_loss.item(), args.step)
            for i, e_i in enumerate(encoding_indices):
                writer.add_scalar('perplexity_{}'.format(i), perplexity(avg_probs(e_i)).item(), args.step)

        return loss


# --------------------
# Train, evaluate, reconstruct
# --------------------

def train_epoch(model, dataloader, optimizer, scheduler, epoch, writer, args):
    model.train()

    with tqdm(total=len(dataloader), desc='epoch {}/{}'.format(epoch, args.start_epoch + args.n_epochs)) as pbar:
        for x, _ in dataloader:
            args.step += 1

            loss = model(x.to(args.device), args.commitment_cost, writer if args.step % args.log_interval == 0 else None)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()

            pbar.set_postfix(loss='{:.4f}'.format(loss.item()))
            pbar.update()

def show_recons_from_hierarchy(model, n_samples, x, z_q, recon_x=None):
    # full reconstruction
    if recon_x is None:
        recon_x = model.decode(None, z_q)
    # top level only reconstruction -- no contribution from bottom-level (level1) latents
    recon_top = model.decode(None, (z_q[0].fill_(0), z_q[1]))

    # construct image grid
    x = make_grid(x[:n_samples].cpu(), normalize=True)
    recon_x = make_grid(recon_x[:n_samples].cpu(), normalize=True)
    recon_top = make_grid(recon_top[:n_samples].cpu(), normalize=True)
    separator = torch.zeros(x.shape[0], 4, x.shape[2])
    return torch.cat([x, separator, recon_x, separator, recon_top], dim=1)

@torch.no_grad()
def evaluate(model, dataloader, args):
    model.eval()

    recon_loss = 0
    for x, _ in tqdm(dataloader):
        x = x.to(args.device)
        z_e = model.encode(x)
        encoding_indices, z_q = model.quantize(z_e)
        recon_x = model.decode(z_e, z_q)
        recon_loss += F.mse_loss(recon_x, x).item()
    recon_loss /= len(dataloader)

    # reconstruct
    recon_image = show_recons_from_hierarchy(model, args.n_samples, x, z_q, recon_x)
    return recon_image, recon_loss

def train_and_evaluate(model, train_dataloader, valid_dataloader, optimizer, scheduler, writer, args):
    for epoch in range(args.start_epoch, args.start_epoch + args.n_epochs):
        train_epoch(model, train_dataloader, optimizer, scheduler, epoch, writer, args)

        # save model
        torch.save({'epoch': epoch,
                    'global_step': args.step,
                    'state_dict': model.state_dict()},
                    os.path.join(args.output_dir, 'checkpoint.pt'))
        torch.save(optimizer.state_dict(), os.path.join(args.output_dir, 'optim_checkpoint.pt'))
        if scheduler: torch.save(optimizer.state_dict(), os.path.join(args.output_dir, 'sched_checkpoint.pt'))

        if (epoch+1) % args.eval_interval == 0:
            # evaluate
            recon_image, recon_loss = evaluate(model, valid_dataloader, args)
            print('Evaluate -- recon loss: {:.4f}'.format(recon_loss))
            writer.add_scalar('loss_recon_eval', recon_loss, args.step)
            writer.add_image('eval_reconstructions', recon_image, args.step)
            save_image(recon_image, os.path.join(args.output_dir, 'eval_reconstruction_step_{}'.format(args.step) + '.png'))


# --------------------
# Main
# --------------------

if __name__ == '__main__':
    args = parser.parse_args()
    if args.restore_dir:
        args.output_dir = args.restore_dir
    if not args.output_dir:  # if not given use results/file_name/time_stamp
        args.output_dir = './results/{}/{}'.format(os.path.splitext(__file__)[0], time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
    writer = SummaryWriter(log_dir = args.output_dir)

    args.device = 'cuda:{}'.format(args.cuda) if args.cuda is not None and torch.cuda.is_available() else 'cpu'

    torch.manual_seed(args.seed)

    # setup dataset and dataloader -- preprocess data to [-1, 1]
    train_dataloader = fetch_vqvae_dataloader(args, train=True)
    valid_dataloader = fetch_vqvae_dataloader(args, train=False)

    # save config
    if not os.path.exists(os.path.join(args.output_dir, 'config_{}.json'.format(args.cuda))):
        save_json(args.__dict__, 'config_{}'.format(args.cuda), args)

    # setup model
    model, optimizer, scheduler = load_model(VQVAE2, args.output_dir, args,
                                             restore=(args.restore_dir is not None),
                                             eval_mode=False,
                                             optimizer_cls=partial(torch.optim.Adam, lr=args.lr),
                                             scheduler_cls=partial(torch.optim.lr_scheduler.ExponentialLR, gamma=args.lr_decay))

    # print and write config with update step and epoch from load_model
    writer.add_text('config', str(args.__dict__), args.step)
    pprint.pprint(args.__dict__)

    if args.train:
        train_and_evaluate(model, train_dataloader, valid_dataloader, optimizer, scheduler, writer, args)

    if args.evaluate:
        recon_image, recon_loss = evaluate(model, valid_dataloader, args)
        print('Evaluate @ step {} -- recon loss: {:.4f}'.format(args.step, recon_loss))
        writer.add_scalar('loss_recon_eval', recon_loss, args.step)
        writer.add_image('eval_reconstructions', recon_image, args.step)
        save_image(recon_image, os.path.join(args.output_dir, 'eval_reconstruction_step_{}'.format(args.step) + '.png'))

