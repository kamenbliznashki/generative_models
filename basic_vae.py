"""
Implementation of Auto-encoding Variational Bayes
https://arxiv.org/pdf/1312.6114.pdf

Reference implementatoin in pytorch examples https://github.com/pytorch/examples/blob/master/vae/
Toy example per Adversarial Variational Bayes https://arxiv.org/abs/1701.04722

"""

import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.utils import make_grid, save_image

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Dataset specific configs for input and latent dimensions.', dest='dataset')

# training params
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=11272018)
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--quiet', action='store_true')

parser.add_argument('--data_dir', default='./data')
parser.add_argument('--output_dir', default='./results/{}'.format(os.path.splitext(__file__)[0]))

# model parameters
toy_subparser = subparsers.add_parser('toy')
toy_subparser.add_argument('--x_dim', type=int, default=4, help='Dimension of the input data.')
toy_subparser.add_argument('--z_dim', type=int, default=2, help='Size of the latent space.')
toy_subparser.add_argument('--hidden_dim', type=int, default=400, help='Size of the hidden layer.')

mnist_subparser = subparsers.add_parser('mnist')
mnist_subparser.add_argument('--x_dim', type=int, default=28*28, help='Dimension of the input data.')
mnist_subparser.add_argument('--z_dim', type=int, default=100, help='Size of the latent space.')
mnist_subparser.add_argument('--hidden_dim', type=int, default=400, help='Size of the hidden layer.')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------
# Data
# --------------------

def fetch_dataloader(args, train=True, download=False):

    transforms = T.Compose([T.ToTensor()])
    dataset = MNIST(root=args.data_dir, train=train, download=download, transform=transforms)

    kwargs = {'num_workers': 1, 'pin_memory': True} if device.type is 'cuda' else {}

    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train, drop_last=True, **kwargs)

class ToyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.x_dim = args.x_dim
        self.batch_size = args.batch_size

    def __len__(self):
        return self.batch_size * 1000

    def __getitem__(self, i):
        one_hot = torch.zeros(self.x_dim)
        label = torch.randint(0, self.x_dim, (1, )).long()
        one_hot[label] = 1.
        return one_hot, label

def fetch_toy_dataloader(args):
    return DataLoader(ToyDataset(args), batch_size=args.batch_size, shuffle=True)


# --------------------
# Plotting helpers
# --------------------

def plot_tsne(model, test_loader, args):
    data = test_loader.dataset.test_data.float() / 255.
    data = data.view(data.shape[0], -1)
    labels = test_loader.dataset.test_labels
    classes = torch.unique(labels, sorted=True).numpy()

    p_x_z, q_z_x = model(data)

    tsne = TSNE(n_components=2, random_state=0)
    z_embed = tsne.fit_transform(q_z_x.loc.cpu().numpy())  # map the posterior mean

    fig = plt.figure()
    for i in classes:
        mask = labels.cpu().numpy() == i
        plt.scatter(z_embed[mask, 0], z_embed[mask, 1], s=10, label=str(i))

    plt.title('Latent variable T-SNE embedding per class')
    plt.legend()
    plt.gca().axis('off')
    fig.savefig(os.path.join(args.output_dir, 'tsne_embedding.png'))


def plot_scatter(model, args):
    data = torch.eye(args.x_dim).repeat(args.batch_size, 1)
    labels = data @ torch.arange(args.x_dim).float()

    _, q_z_x = model(data)
    z = q_z_x.sample().numpy()
    plt.scatter(z[:,0], z[:,1], c=labels.data.numpy(), alpha=0.5)

    plt.title('Latent space embedding per class\n(n_iter = {})'.format(len(ToyDataset(args))*args.n_epochs))
    plt.savefig(os.path.join(args.output_dir, 'latent_distribution_toy_example.png'))
    plt.close()

# --------------------
# Model
# --------------------

class VAE(nn.Module):
    def __init__(self, args):#in_dim=784, hidden_dim=400, z_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(args.x_dim, args.hidden_dim)
        self.fc21 = nn.Linear(args.hidden_dim, args.z_dim)
        self.fc22 = nn.Linear(args.hidden_dim, args.z_dim)
        self.fc3 = nn.Linear(args.z_dim, args.hidden_dim)
        self.fc4 = nn.Linear(args.hidden_dim, args.x_dim)

    # q(z|x) parametrizes the approximate posterior as a Normal(mu, scale)
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        scale = self.fc22(h1).exp()
        return D.Normal(mu, scale)

    # p(x|z) returns the likelihood of data given the latents
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        logits = self.fc4(h3)
        return D.Bernoulli(logits=logits)

    def forward(self, x):
        q_z_x = self.encode(x.view(x.shape[0], -1))     # returns Normal
        p_x_z = self.decode(q_z_x.rsample())            # returns Bernoulli; note reparametrization when sampling the approximate
        return p_x_z, q_z_x


# ELBO loss
def loss_fn(p_x_z, q_z_x, x):
    # Equation 3 from Kingma & Welling -- Auto-Encoding Variational Bayes
    #   ELBO = - KL( q(z|x), p(z) ) + Expectation_under_q(z|x)_[log p(x|z)]
    #   this simplifies to eq 7 from Kingma nad Welling where the expectation is avg of z samples
    #   signs are revered from paper as paper maximizes ELBO and here we min - ELBO
    #   both KLD and BCE are summed over dim 1 (image H*W) and mean over dim 0 (batch)
    p_z = D.Normal(torch.FloatTensor([0], device=x.device), torch.FloatTensor([1], device=x.device))
    KLD = D.kl.kl_divergence(q_z_x, p_z).sum(1).mean(0)             # divergene of the approximate posterior from the prior
    BCE = - p_x_z.log_prob(x.view(x.shape[0], -1)).sum(1).mean(0)   # expected negative reconstruction error;
                                                                    # prob density of data x under the generative model given by z
    return BCE + KLD


# --------------------
# Train and eval
# --------------------

def train_epoch(model, dataloader, loss_fn, optimizer, epoch, args):
    model.train()

    ELBO_loss = 0

    with tqdm(total=len(dataloader), desc='epoch {} of {}'.format(epoch+1, args.n_epochs)) as pbar:
        for i, (data, _) in enumerate(dataloader):
            data = data.to(device)

            p_x_z, q_z_x = model(data)
            loss = loss_fn(p_x_z, q_z_x, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update tracking
            pbar.set_postfix(loss='{:.3f}'.format(loss.item()))
            pbar.update()

            ELBO_loss += loss.item()

    print('Epoch: {} Average ELBO loss: {:.4f}'.format(epoch+1, ELBO_loss / (len(dataloader))))


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, epoch, args):
    model.eval()

    ELBO_loss = 0

    with tqdm(total=len(dataloader)) as pbar:
        for i, (data, _) in enumerate(dataloader):
            data = data.to(device)
            p_x_z, q_z_x = model(data)

            ELBO_loss += loss_fn(p_x_z, q_z_x, data).item()

            pbar.update()

            if i == 0 and args.dataset == 'mnist':
                nrow = 10
                n = min(data.size(0), nrow**2)
                real_data = make_grid(data[:n].cpu(), nrow)
                spacer = torch.ones(real_data.shape[0], real_data.shape[1], 5)
                generated_data = make_grid(p_x_z.probs.view(args.batch_size, 1, 28, 28)[:n].cpu(), nrow)
                image = torch.cat([real_data, spacer, generated_data], dim=-1)
                save_image(image, os.path.join(args.output_dir, 'reconstruction_at_epoch_' + str(epoch) + '.png'), nrow)

    print('Test set average ELBO loss: {:.4f}'.format(ELBO_loss / len(dataloader)))


def train_and_evaluate(model, train_loader, test_loader, loss_fn, optimizer, args):
    for epoch in range(args.n_epochs):
        train_epoch(model, train_loader, loss_fn, optimizer, epoch, args)
        evaluate(model, test_loader, loss_fn, epoch, args)

        # save weights
        if args.save_model:
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'vae_model_xdim{}_hdim{}_zdim{}.pt'.format(
                                                                            args.x_dim, args.hidden_dim, args.z_dim)))

        # show samples
        if args.dataset == 'mnist':
            with torch.no_grad():
                # sample p(z) = Normal(0, 1)
                prior_sample = torch.randn(64, args.z_dim).to(device)
                # compute likelihood p(x|z) decoder; returns torch.distribution.Bernoulli
                likelihood = model.decode(prior_sample).probs
                save_image(likelihood.cpu().view(64, 1, 28, 28), os.path.join(args.output_dir, 'sample_at_epoch_' + str(epoch) + '.png'))



if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.isdir(os.path.join(args.output_dir, args.dataset)):
        os.makedirs(os.path.join(args.output_dir, args.dataset))
    args.output_dir = os.path.join(args.output_dir, args.dataset)

    torch.manual_seed(args.seed)

    # data
    if args.dataset == 'toy':
        train_loader = fetch_toy_dataloader(args)
        test_loader = train_loader
    else:
        train_loader = fetch_dataloader(args, train=True)
        test_loader = fetch_dataloader(args, train=False)

    # model
    model = VAE(args).to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train and eval
    train_and_evaluate(model, train_loader, test_loader, loss_fn, optimizer, args)

    # visualize z space
    with torch.no_grad():
        if args.dataset == 'toy':
            plot_scatter(model, args)
        else:
            pass
            plot_tsne(model, test_loader, args)

