"""
InfoGAN -- https://arxiv.org/abs/1606.03657

Follows the Tensorflow implementation at http://www.depthfirstlearning.com/2018/InfoGAN

"""


import os
import argparse
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid

import utils

parser = argparse.ArgumentParser()

# training params
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--noise_dim', type=int, default=62, help='Size of the categorical latent representation')
parser.add_argument('--cat_dim', type=int, default=10, help='Size of the categorical latent representation')
parser.add_argument('--cont_dim', type=int, default=2, help='Size of the continuous latent representation')
parser.add_argument('--info_reg_coeff', default=1., help='The weight of the MI regularization hyperparameter')
parser.add_argument('--g_lr', default=1e-3, help='Generator learning rate')
parser.add_argument('--d_lr', default=2e-4, help='Discriminator learning rate')
parser.add_argument('--log_interval', default=100)
parser.add_argument('--cuda', type=int, help='Which cuda device to use')
parser.add_argument('--mini_data', action='store_true')
# eval params
parser.add_argument('--evaluate_on_grid', action='store_true')
# data paths
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--data_dir', default='./data')
parser.add_argument('--output_dir', default='./results/infogan')
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
    """ base for the Discriminator (D) and latent recognition network (Q) """
    def __init__(self):
        super().__init__()
        # base network shared between discriminator D and recognition network Q
        self.base_net = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),                 # out (B, 64, 14, 14)
                                      nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),   # out (B, 128, 7, 7)
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU(0.1, True),
                                      Flatten(),
                                      nn.Linear(128*7*7, 1024, bias=False),
                                      nn.BatchNorm1d(1024),
                                      nn.LeakyReLU(0.1, True))

        # discriminator -- real vs fake binary output
        self.d = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.base_net(x).squeeze()
        logits_real = self.d(x)
        # return feature representation and real vs fake prob
        return x, dist.Bernoulli(logits=logits_real)


class Q(nn.Module):
    """ Latent space recognition network; shares base network of the discriminator """
    def __init__(self, cat_dim, cont_dim, fix_cont_std=True):
        super().__init__()
        self.cat_dim = cat_dim
        self.cont_dim = cont_dim
        self.fix_cont_std = fix_cont_std

        # recognition network for latent vars ie encoder, shared between the factors of q
        self.encoder = nn.Sequential(nn.Linear(1024, 128, bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(0.1, True))

        # the factors of q -- 1 categorical and 2 continuous variables
        self.q = nn.Linear(128, cat_dim + 2 * cont_dim)

    def forward(self, x):
        # latent space encoding
        z = self.encoder(x)

        logits_cat, cont_mu, cont_var = torch.split(self.q(z), [self.cat_dim, self.cont_dim, self.cont_dim], dim=-1)

        if self.fix_cont_std:
            cont_sigma = torch.ones_like(cont_mu)
        else:
            cont_sigma = F.softplus(cont_var)

        q_cat = dist.Categorical(logits=logits_cat)
        q_cont = dist.Normal(loc=cont_mu, scale=cont_sigma)

        return q_cat, q_cont


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(74, 1024, bias=False),
                                 nn.BatchNorm1d(1024),
                                 nn.ReLU(True),
                                 nn.Linear(1024, 7*7*128),
                                 nn.BatchNorm1d(7*7*128),
                                 nn.ReLU(True),
                                 Unflatten(-1, 128, 7, 7),
                                 nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(True),
                                 nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
                                 nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


def initialize_weights(m):
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.normal_(mean=1., std=0.02)
        m.bias.data.fill_(0.)
    else:
        try:
            m.weight.data.normal_(std=0.02)
        except AttributeError:  # skip activation layers
            pass


# --------------------
# Train
# --------------------

def sample_z(args):
    # generate samples from the prior
    z_cat = OneHotCategorical(logits=torch.zeros(args.batch_size, args.cat_dim)).sample()
    z_noise = dist.Uniform(-1, 1).sample(torch.Size((args.batch_size, args.noise_dim)))
    z_cont = dist.Uniform(-1, 1).sample(torch.Size((args.batch_size, args.cont_dim)))

    # concatenate the incompressible noise, discrete latest, and continuous latents
    z = torch.cat([z_noise, z_cat, z_cont], dim=1)

    return z.to(args.device), z_cat.to(args.device), z_noise.to(args.device), z_cont.to(args.device)


def info_loss_fn(cat_fake, cont_fake, z_cat, z_cont, args):
    log_prob_cat = cat_fake.log_prob(z_cat.nonzero()[:,1]).mean()   # equivalent to pytorch cross_entropy loss fn
    log_prob_cont = cont_fake.log_prob(z_cont).sum(1).mean()

    info_loss = - args.info_reg_coeff * (log_prob_cat + log_prob_cont)
    return log_prob_cat, log_prob_cont, info_loss



def train_epoch(D, Q, G, dataloader, d_optimizer, g_optimizer, epoch, writer, args):

    fixed_z, _, _, _ = sample_z(args)

    real_labels = torch.ones(args.batch_size, 1, device=args.device).requires_grad_(False)
    fake_labels = torch.zeros(args.batch_size, 1, device=args.device).requires_grad_(False)

    with tqdm(total=len(dataloader), desc='epoch {} of {}'.format(epoch+1, args.n_epochs)) as pbar:
        time.sleep(0.1)

        for i, (x, _) in enumerate(dataloader):
            D.train()
            G.train()

            x = x.to(args.device)
#            x = 2*x - 0.5


            # train Generator
            z, z_cat, z_noise, z_cont = sample_z(args)

            generated = G(z)
            x_pre_q, d_fake = D(generated)
            q_cat, q_cont = Q(x_pre_q)

            gan_g_loss = - d_fake.log_prob(real_labels).mean()  # equivalent to pytorch binary_cross_entropy_with_logits loss fn
            log_prob_cat, log_prob_cont, info_loss = info_loss_fn(q_cat, q_cont, z_cat, z_cont, args)

            g_loss = gan_g_loss + info_loss

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()


            # train Discriminator
            _, d_real = D(x)
            x_pre_q, d_fake = D(generated.detach())
            q_cat, q_cont = Q(x_pre_q)

            gan_d_loss = - d_real.log_prob(real_labels).mean() - d_fake.log_prob(fake_labels).mean()
            log_prob_cat, log_prob_cont, info_loss = info_loss_fn(q_cat, q_cont, z_cat, z_cont, args)

            d_loss = gan_d_loss + info_loss

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()


            # update tracking
            pbar.set_postfix(log_prob_cat='{:.3f}'.format(log_prob_cat.item()),
                             log_prob_cont='{:.3f}'.format(log_prob_cont.item()),
                             d_loss='{:.3f}'.format(gan_d_loss.item()),
                             g_loss='{:.3f}'.format(gan_g_loss.item()),
                             i_loss='{:.3f}'.format(info_loss.item()))
            pbar.update()

            if i % args.log_interval == 0:
                step = epoch
                writer.add_scalar('gan_d_loss', gan_d_loss.item(), step)
                writer.add_scalar('gan_g_loss', gan_g_loss.item(), step)
                writer.add_scalar('info_loss', info_loss.item(), step)
                writer.add_scalar('log_prob_cat', log_prob_cat.item(), step)
                writer.add_scalar('log_prob_cont', log_prob_cont.item(), step)
                # sample images
                with torch.no_grad():
                    G.eval()
                    fake_images = G(fixed_z)
                    writer.add_image('generated', make_grid(fake_images[:10].cpu(), nrow=10, normalize=True, padding=1), step)
                    save_image(fake_images[:10].cpu(),
                               os.path.join(args.output_dir, 'generated_sample_epoch_{}.png'.format(epoch)),
                               nrow=10)


def train(D, Q, G, dataloader, d_optimizer, g_optimizer, writer, args):

    print('Starting training with args:\n', args)

    start_epoch = 0

    if args.restore_file:
        print('Restoring parameters from {}'.format(args.restore_file))
        start_epoch = utils.load_checkpoint(args.restore_file, [D, Q, G], [d_optimizer, g_optimizer], map_location=args.device.type)
        args.n_epochs += start_epoch - 1
        print('Resuming training from epoch {}'.format(start_epoch))

    for epoch in range(start_epoch, args.n_epochs):
        train_epoch(D, Q, G, dataloader, d_optimizer, g_optimizer, epoch, writer, args)

        # snapshot at end of epoch
        if args.save_model:
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'model_state_dicts': [D.state_dict(), Q.state_dict(), G.state_dict()],
                                   'optimizer_state_dicts': [d_optimizer.state_dict(), g_optimizer.state_dict()]},
                                   checkpoint=args.output_dir,
                                   quiet=True)

# --------------------
# Evaluate
# --------------------

@torch.no_grad()
def evaluate_on_grid(G, writer, args):
    # sample noise randomly
    z_noise = torch.empty(100, args.noise_dim).uniform_(-1,1)
    # order the categorical latent
    z_cat = torch.eye(10).repeat(10,1)
    # order the first continuous latent
    c = torch.linspace(-2, 2, 10).view(-1,1).repeat(1,10).reshape(-1,1)
    z_cont = torch.cat([c, torch.zeros_like(c)], dim=1).reshape(100, 2)

    # combine into z and pass through generator
    z = torch.cat([z_noise, z_cat, z_cont], dim=1).to(args.device)
    fake_images = G(z)
    writer.add_image('c1 cont generated', make_grid(fake_images.cpu(), nrow=10, normalize=True, padding=1))
    save_image(fake_images.cpu(),
               os.path.join(args.output_dir, 'latent_var_grid_sample_c1.png'),
               nrow=10)

    # order second continuous latent; combine into z and pass through generator
    z_cont = z_cont.flip(1)
    z = torch.cat([z_noise, z_cat, z_cont], dim=1).to(args.device)
    fake_images = G(z)
    writer.add_image('c2 cont generated', make_grid(fake_images.cpu(), nrow=10, normalize=True, padding=1))
    save_image(fake_images.cpu(),
               os.path.join(args.output_dir, 'latent_var_grid_sample_c2.png'),
               nrow=10)


# --------------------
# Run
# --------------------

if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    writer = utils.set_writer(args.output_dir, '_train')

    args.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda is not None else 'cpu')

    # set seed
    torch.manual_seed(11122018)
    if args.device.type is 'cuda': torch.cuda.manual_seed(11122018)

    # models
    D = Discriminator().to(args.device)
    Q = Q(args.cat_dim, args.cont_dim).to(args.device)
    G = Generator().to(args.device)
    D.apply(initialize_weights)
    Q.apply(initialize_weights)
    G.apply(initialize_weights)

    # optimizers
    g_optimizer = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam([{'params': D.parameters()},
                                    {'params': Q.parameters()}], lr=args.d_lr, betas=(0.5, 0.999))

    # eval
    if args.evaluate_on_grid:
        print('Restoring parameters from {}'.format(args.restore_file))
        _ = utils.load_checkpoint(args.restore_file, [D, Q, G], [d_optimizer, g_optimizer])
        evaluate_on_grid(G, writer, args)
    # train
    else:
        dataloader = fetch_dataloader(args)
        train(D, Q, G, dataloader, d_optimizer, g_optimizer, writer, args)
        evaluate_on_grid(G, writer, args)

    writer.close()
