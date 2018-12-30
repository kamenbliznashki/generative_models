"""
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
https://arxiv.org/pdf/1511.06434

Notes:
    Model architecture differs from paper:
        generator ends with Sigmoid
        inputs normalized to [0,1]
        learning rates differ

"""


import os
import argparse
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid

import utils

parser = argparse.ArgumentParser()

# training params
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--noise_dim', type=int, default=96, help='Size of the latent representation.')
parser.add_argument('--g_lr', type=float, default=1e-3, help='Generator learning rate')
parser.add_argument('--d_lr', type=float, default=1e-4, help='Discriminator learning rate')
parser.add_argument('--log_interval', default=100)
parser.add_argument('--cuda', type=int, help='Which cuda device to use')
parser.add_argument('--mini_data', action='store_true')
# eval params
parser.add_argument('--evaluate_on_grid', action='store_true')
# data paths
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--data_dir', default='./data')
parser.add_argument('--output_dir', default='./results/dcgan')
parser.add_argument('--restore_file', help='Path to .pt checkpoint file for Discriminator and Generator')




# --------------------
# Data
# --------------------

def fetch_dataloader(args, train=True, download=True, mini_size=128):
    # load dataset and init in the dataloader

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

    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=train, drop_last=True, **kwargs)

    return dl


# --------------------
# Model
# --------------------

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Unflatten(nn.Module):
    def __init__(self, B, C, H, W):
        super().__init__()
        self.B = B
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.reshape(self.B, self.C, self.H, self.W)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),      # out (B, 64, 14, 14)
                                 nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),    # out (B, 128, 7, 7)
                                 nn.BatchNorm2d(128),
                                 nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0, bias=False),   # out (B, 128, 4, 4)
                                 nn.BatchNorm2d(256),
                                 nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(256, 512, kernel_size=4, bias=False),                        # out (B, 256, 1, 1)
                                 nn.BatchNorm2d(512),
                                 nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(512, 1, kernel_size=1, bias=False))

    def forward(self, x):
        return dist.Bernoulli(logits=self.net(x).squeeze())


class Generator(nn.Module):
    def __init__(self, noise_dim):
        super().__init__()
        self.net = nn.Sequential(nn.ConvTranspose2d(noise_dim, 512, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(True),
                                 nn.ConvTranspose2d(512, 256, kernel_size=4, bias=False),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU(True),
                                 nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(True),
                                 nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(True),
                                 nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
                                 nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


def initialize_weights(m, std=0.02):
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.normal_(mean=1., std=std)
        m.bias.data.fill_(0.)
    else:
        try:
            m.weight.data.normal_(std=std)
        except AttributeError:  # skip activation layers
            pass



# --------------------
# Train
# --------------------

def sample_z(args):
    # generate samples from the prior
    return dist.Uniform(-1,1).sample((args.batch_size, args.noise_dim, 1, 1)).to(args.device)


def train_epoch(D, G, dataloader, d_optimizer, g_optimizer, epoch, writer, args):

    fixed_z = sample_z(args)

    real_labels = torch.ones(args.batch_size, 1, device=args.device).requires_grad_(False)
    fake_labels = torch.zeros(args.batch_size, 1, device=args.device).requires_grad_(False)

    with tqdm(total=len(dataloader), desc='epoch {} of {}'.format(epoch+1, args.n_epochs)) as pbar:
        time.sleep(0.1)

        for i, (x, _) in enumerate(dataloader):
            D.train()
            G.train()

            x = x.to(args.device)

            # train generator

            # sample prior
            z = sample_z(args)

            # run through model
            generated = G(z)
            d_fake = D(generated)

            # calculate losses
            g_loss = - d_fake.log_prob(real_labels).mean()

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()


            # train discriminator
            d_real = D(x)
            d_fake = D(generated.detach())

            # calculate losses
            d_loss = - d_real.log_prob(real_labels).mean() - d_fake.log_prob(fake_labels).mean()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()


            # update tracking
            pbar.set_postfix(d_loss='{:.3f}'.format(d_loss.item()),
                             g_loss='{:.3f}'.format(g_loss.item()))
            pbar.update()

            if i % args.log_interval == 0:
                step = epoch
                writer.add_scalar('d_loss', d_loss.item(), step)
                writer.add_scalar('g_loss', g_loss.item(), step)
                # sample images
                with torch.no_grad():
                    G.eval()
                    fake_images = G(fixed_z)
                    writer.add_image('generated', make_grid(fake_images[:10].cpu(), nrow=10, padding=1), step)
                    save_image(fake_images[:10].cpu(),
                               os.path.join(args.output_dir, 'generated_sample_epoch_{}.png'.format(epoch)),
                               nrow=10)


def train(D, G, dataloader, d_optimizer, g_optimizer, writer, args):

    print('Starting training with args:\n', args)

    start_epoch = 0

    if args.restore_file:
        print('Restoring parameters from {}'.format(args.restore_file))
        start_epoch = utils.load_checkpoint(args.restore_file, [D, G], [d_optimizer, g_optimizer])
        args.n_epochs += start_epoch - 1
        print('Resuming training from epoch {}'.format(start_epoch))

    for epoch in range(start_epoch, args.n_epochs):
        train_epoch(D, G, dataloader, d_optimizer, g_optimizer, epoch, writer, args)

        # snapshot at end of epoch
        if args.save_model:
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'model_state_dicts': [D.state_dict(), G.state_dict()],
                                   'optimizer_state_dicts': [d_optimizer.state_dict(), g_optimizer.state_dict()]},
                                   checkpoint=args.output_dir,
                                   quiet=True)

@torch.no_grad()
def evaluate_on_grid(G, writer, args):
    # sample noise randomly
    z = torch.empty(100, args.noise_dim, 1, 1).uniform_(-1,1).to(args.device)

    fake_images = G(z)
    writer.add_image('generated grid', make_grid(fake_images.cpu(), nrow=10, normalize=True, padding=1))
    save_image(fake_images.cpu(),
               os.path.join(args.output_dir, 'latent_var_grid_sample_c1.png'),
               nrow=10)



if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    writer = utils.set_writer(args.output_dir, '_train')

    args.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda is not None else 'cpu')

    # set seed
    torch.manual_seed(11122018)
    if args.device is torch.device('cuda'): torch.cuda.manual_seed(11122018)

    # input
    dataloader = fetch_dataloader(args)

    # models
    D = Discriminator().to(args.device)
    G = Generator(args.noise_dim).to(args.device)
    D.apply(initialize_weights)
    G.apply(initialize_weights)

    # optimizers
    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))

    # train
    # eval
    if args.evaluate_on_grid:
        print('Restoring parameters from {}'.format(args.restore_file))
        _ = utils.load_checkpoint(args.restore_file, [D, G], [d_optimizer, g_optimizer])
        evaluate_on_grid(G, writer, args)
    # train
    else:
        dataloader = fetch_dataloader(args)
        train(D, G, dataloader, d_optimizer, g_optimizer, writer, args)
        evaluate_on_grid(G, writer, args)
    writer.close()

