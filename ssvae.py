"""
Semi-supervised Learning with Deep Generative Models
https://arxiv.org/pdf/1406.5298.pdf
"""

import os
import argparse
from tqdm import tqdm
import pprint
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid

parser = argparse.ArgumentParser()

# actions
parser.add_argument('--train', action='store_true', help='Train a new or restored model.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate a model.')
parser.add_argument('--generate', action='store_true', help='Generate samples from a model.')
parser.add_argument('--vis_styles', action='store_true', help='Visualize styles manifold.')
parser.add_argument('--cuda', type=int, help='Which cuda device to use')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')

# file paths
parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
parser.add_argument('--data_dir', default='./data/', help='Location of dataset.')
parser.add_argument('--output_dir', default='./results/{}'.format(os.path.splitext(__file__)[0]))
parser.add_argument('--results_file', default='results.txt', help='Filename where to store settings and test results.')

# model parameters
parser.add_argument('--image_dims', type=tuple, default=(1,28,28), help='Dimensions of a single datapoint (e.g. (1,28,28) for MNIST).')
parser.add_argument('--z_dim', type=int, default=50, help='Size of the latent representation.')
parser.add_argument('--y_dim', type=int, default=10, help='Size of the labels / output.')
parser.add_argument('--hidden_dim', type=int, default=500, help='Size of the hidden layer.')

# training params
parser.add_argument('--n_labeled', type=int, default=3000, help='Number of labeled training examples in the dataset')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--alpha', type=float, default=0.1, help='Classifier loss multiplier controlling generative vs. discriminative learning.')


# --------------------
# Data
# --------------------

# create semi-supervised datasets of labeled and unlabeled data with equal number of labels from each class
def create_semisupervised_datasets(dataset, n_labeled):
    # note this is only relevant for training the model
    assert dataset.train == True, 'Dataset must be the training set; assure dataset.train = True.'

    # compile new x and y and replace the dataset.train_data and train_labels with the 
    x = dataset.train_data
    y = dataset.train_labels
    n_x = x.shape[0]
    n_classes = len(torch.unique(y))

    assert n_labeled % n_classes == 0, 'n_labeld not divisible by n_classes; cannot assure class balance.'
    n_labeled_per_class = n_labeled // n_classes

    x_labeled = [0] * n_classes
    x_unlabeled = [0] * n_classes
    y_labeled = [0] * n_classes
    y_unlabeled = [0] * n_classes

    for i in range(n_classes):
        idxs = (y == i).nonzero().data.numpy()
        np.random.shuffle(idxs)

        x_labeled[i] = x[idxs][:n_labeled_per_class]
        y_labeled[i] = y[idxs][:n_labeled_per_class]
        x_unlabeled[i] = x[idxs][n_labeled_per_class:]
        y_unlabeled[i] = y[idxs][n_labeled_per_class:]

    # construct new labeled and unlabeled datasets
    labeled_dataset = copy.deepcopy(dataset)
    labeled_dataset.train_data = torch.cat(x_labeled, dim=0).squeeze()
    labeled_dataset.train_labels = torch.cat(y_labeled, dim=0)

    unlabeled_dataset = copy.deepcopy(dataset)
    unlabeled_dataset.train_data = torch.cat(x_unlabeled, dim=0).squeeze()
    unlabeled_dataset.train_labels = torch.cat(y_unlabeled, dim=0)

    del dataset

    return labeled_dataset, unlabeled_dataset


def fetch_dataloaders(args):
    assert args.n_labeled != None, 'Must provide n_labeled number to split dataset.'

    transforms = T.Compose([T.ToTensor()])
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.device.type is 'cuda' else {}

    def get_dataset(train):
        return MNIST(root=args.data_dir, train=train, transform=transforms)

    def get_dl(dataset):
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=dataset.train, drop_last=True, **kwargs)

    test_dataset = get_dataset(train=False)
    train_dataset = get_dataset(train=True)
    labeled_dataset, unlabeled_dataset = create_semisupervised_datasets(train_dataset, args.n_labeled)

    return get_dl(labeled_dataset), get_dl(unlabeled_dataset), get_dl(test_dataset)


def one_hot(x, label_size):
    out = torch.zeros(len(x), label_size).to(x.device)
    out[torch.arange(len(x)), x.squeeze()] = 1
    return out

# --------------------
# Model
# --------------------

class SSVAE(nn.Module):
    """
    Data model (SSL paper eq 2):
        p(y) = Cat(y|pi)
        p(z) = Normal(z|0,1)
        p(x|y,z) = f(x; z,y,theta)

    Recognition model / approximate posterior q_phi (SSL paper eq 4):
        q(y|x) = Cat(y|pi_phi(x))
        q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x)))


    """
    def __init__(self, args):
        super().__init__()
        C, H, W = args.image_dims
        x_dim = C * H * W

        # --------------------
        # p model -- SSL paper generative semi supervised model M2
        # --------------------

        self.p_y = D.OneHotCategorical(probs=1 / args.y_dim * torch.ones(1,args.y_dim, device=args.device))
        self.p_z = D.Normal(torch.tensor(0., device=args.device), torch.tensor(1., device=args.device))

        # parametrized data likelihood p(x|y,z)
        self.decoder = nn.Sequential(nn.Linear(args.z_dim + args.y_dim, args.hidden_dim),
                                     nn.Softplus(),
                                     nn.Linear(args.hidden_dim, args.hidden_dim),
                                     nn.Softplus(),
                                     nn.Linear(args.hidden_dim, x_dim))

        # --------------------
        # q model -- SSL paper eq 4
        # --------------------

        # parametrized q(y|x) = Cat(y|pi_phi(x)) -- outputs parametrization of categorical distribution
        self.encoder_y = nn.Sequential(nn.Linear(x_dim, args.hidden_dim),
                                       nn.Softplus(),
                                       nn.Linear(args.hidden_dim, args.hidden_dim),
                                       nn.Softplus(),
                                       nn.Linear(args.hidden_dim, args.y_dim))

        # parametrized q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x))) -- output parametrizations for mean and diagonal variance of a Normal distribution
        self.encoder_z = nn.Sequential(nn.Linear(x_dim + args.y_dim, args.hidden_dim),
                                       nn.Softplus(),
                                       nn.Linear(args.hidden_dim, args.hidden_dim),
                                       nn.Softplus(),
                                       nn.Linear(args.hidden_dim, 2 * args.z_dim))


        # initialize weights to N(0, 0.001) and biases to 0 (cf SSL section 4.4)
        for p in self.parameters():
            p.data.normal_(0, 0.001)
            if p.ndimension() == 1: p.data.fill_(0.)

    # q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x))) -- SSL paper eq 4
    def encode_z(self, x, y):
        xy = torch.cat([x, y], dim=1)
        mu, logsigma = self.encoder_z(xy).chunk(2, dim=-1)
        return D.Normal(mu, logsigma.exp())

    # q(y|x) = Categorical(y|pi_phi(x)) -- SSL paper eq 4
    def encode_y(self, x):
        return D.OneHotCategorical(logits=self.encoder_y(x))

    # p(x|y,z) = Bernoulli
    def decode(self, y, z):
        yz = torch.cat([y,z], dim=1)
        return D.Bernoulli(logits=self.decoder(yz))

    # classification model q(y|x) using the trained q distribution
    def forward(self, x):
        y_probs = self.encode_y(x).probs
        return y_probs.max(dim=1)[1]  # return pred labels = argmax


def loss_components_fn(x, y, z, p_y, p_z, p_x_yz, q_z_xy):
    # SSL paper eq 6 for an given y (observed or enumerated from q_y)
    return - p_x_yz.log_prob(x).sum(1) \
           - p_y.log_prob(y) \
           - p_z.log_prob(z).sum(1) \
           + q_z_xy.log_prob(z).sum(1)


# --------------------
# Train and eval
# --------------------

def train_epoch(model, labeled_dataloader, unlabeled_dataloader, loss_components_fn, optimizer, epoch, args):
    model.train()

    n_batches = len(labeled_dataloader) + len(unlabeled_dataloader)
    n_unlabeled_per_labeled = len(unlabeled_dataloader) // len(labeled_dataloader) + 1

    labeled_dataloader = iter(labeled_dataloader)
    unlabeled_dataloader = iter(unlabeled_dataloader)

    with tqdm(total=n_batches, desc='epoch {} of {}'.format(epoch+1, args.n_epochs)) as pbar:
        for i in range(n_batches):
            is_supervised = i % n_unlabeled_per_labeled == 0

            # get batch from respective dataloader
            if is_supervised:
                x, y = next(labeled_dataloader)
                y = one_hot(y, args.y_dim).to(args.device)
            else:
                x, y = next(unlabeled_dataloader)
                y = None
            x = x.to(args.device).view(x.shape[0], -1)

            # compute loss -- SSL paper eq 6, 7, 9
            q_y = model.encode_y(x)
            # labeled data loss -- SSL paper eq 6 and eq 9
            if y is not None:
                q_z_xy = model.encode_z(x, y)
                z = q_z_xy.rsample()
                p_x_yz = model.decode(y, z)
                loss = loss_components_fn(x, y, z, model.p_y, model.p_z, p_x_yz, q_z_xy)
                loss -= args.alpha * args.n_labeled * q_y.log_prob(y)  # SSL eq 9
            # unlabeled data loss -- SSL paper eq 7
            else:
                # marginalize y according to q_y
                loss = - q_y.entropy()
                for y in q_y.enumerate_support():
                    q_z_xy = model.encode_z(x, y)
                    z = q_z_xy.rsample()
                    p_x_yz = model.decode(y, z)
                    L_xy = loss_components_fn(x, y, z, model.p_y, model.p_z, p_x_yz, q_z_xy)
                    loss += q_y.log_prob(y).exp() * L_xy
            loss = loss.mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update trackers
            pbar.set_postfix(loss='{:.3f}'.format(loss.item()))
            pbar.update()


@torch.no_grad()
def evaluate(model, dataloader, epoch, args):
    model.eval()

    accurate_preds = 0

    with tqdm(total=len(dataloader), desc='eval') as pbar:
        for i, (x, y) in enumerate(dataloader):
            x = x.to(args.device).view(x.shape[0], -1)
            y = y.to(args.device)
            preds = model(x)

            accurate_preds += (preds == y).sum().item()

            pbar.set_postfix(accuracy='{:.3f}'.format(accurate_preds / ((i+1) * args.batch_size)))
            pbar.update()

    output = (epoch != None)*'Epoch {} -- '.format(epoch) + 'Test set accuracy: {:.3f}'.format(accurate_preds / (args.batch_size * len(dataloader)))
    print(output)
    print(output, file=open(args.results_file, 'a'))


def train_and_evaluate(model, labeled_dataloader, unlabeled_dataloader, test_dataloader, loss_components_fn, optimizer, args):
    for epoch in range(args.n_epochs):
        train_epoch(model, labeled_dataloader, unlabeled_dataloader, loss_components_fn, optimizer, epoch, args)
        evaluate(model, test_dataloader, epoch, args)

        # save weights
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'ssvae_model_state_hdim{}_zdim{}.pt'.format(
                                                                        args.hidden_dim, args.z_dim)))

        # show samples -- SSL paper Figure 1-b
        generate(model, test_dataloader.dataset, args, epoch)


# --------------------
# Visualize
# --------------------

@torch.no_grad()
def generate(model, dataset, args, epoch=None, n_samples=10):
    n_samples_per_label = 10

    # some interesting samples per paper implementation
    idxs = [7910, 8150, 3623, 2645, 4066, 9660, 5083, 948, 2595, 2]

    x = torch.stack([dataset[i][0] for i in idxs], dim=0).to(args.device)
    y = torch.stack([dataset[i][1] for i in idxs], dim=0).to(args.device)
    y = one_hot(y, args.y_dim)

    q_z_xy = model.encode_z(x.view(n_samples_per_label, -1), y)
    z = q_z_xy.loc
    z = z.repeat(args.y_dim, 1, 1).transpose(0, 1).contiguous().view(-1, args.z_dim)

    # hold z constant and vary y:
    y = torch.eye(args.y_dim).repeat(n_samples_per_label, 1).to(args.device)
    generated_x = model.decode(y, z).probs.view(n_samples_per_label, args.y_dim, *args.image_dims)
    generated_x = generated_x.contiguous().view(-1, *args.image_dims)  # out (n_samples * n_label, C, H, W)

    x = make_grid(x.cpu(), nrow=1)
    spacer = torch.ones(x.shape[0], x.shape[1], 5)
    generated_x = make_grid(generated_x.cpu(), nrow=args.y_dim)
    image = torch.cat([x, spacer, generated_x], dim=-1)
    save_image(image,
               os.path.join(args.output_dir, 'analogies_sample' + (epoch != None)*'_at_epoch_{}'.format(epoch) + '.png'),
               nrow=args.y_dim)


@torch.no_grad()
def vis_styles(model, args):
    assert args.z_dim == 2, 'Style viualization requires z_dim=2'

    for y in range(2,5):
        y = one_hot(torch.tensor(y).unsqueeze(-1), args.y_dim).expand(100, args.y_dim).to(args.device)

        # order the first dim of the z latent
        c = torch.linspace(-5, 5, 10).view(-1,1).repeat(1,10).reshape(-1,1)
        z = torch.cat([c, torch.zeros_like(c)], dim=1).reshape(100, 2).to(args.device)

        # combine into z and pass through decoder
        x = model.decode(y, z).probs.view(y.shape[0], *args.image_dims)
        save_image(x.cpu(),
                   os.path.join(args.output_dir, 'latent_var_grid_sample_c1_y{}.png'.format(y[0].nonzero().item())),
                   nrow=10)

        # order second dim of latent and pass through decoder
        z = z.flip(1)
        x = model.decode(y, z).probs.view(y.shape[0], *args.image_dims)
        save_image(x.cpu(),
                   os.path.join(args.output_dir, 'latent_var_grid_sample_c2_y{}.png'.format(y[0].nonzero().item())),
                   nrow=10)


# --------------------
# Main
# --------------------

if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    args.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda != None else 'cpu')
    torch.manual_seed(args.seed)
    if args.device.type == 'cuda': torch.cuda.manual_seed(args.seed)

    # dataloaders
    labeled_dataloader, unlabeled_dataloader, test_dataloader = fetch_dataloaders(args)

    # model
    model = SSVAE(args).to(args.device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.restore_file:
        # load model and optimizer states
        state = torch.load(args.restore_file, map_location=args.device)
        model.load_state_dict(state)
        # set up paths
        args.output_dir = os.path.dirname(args.restore_file)
    args.results_file = os.path.join(args.output_dir, args.results_file)

    print('Loaded settings and model:')
    print(pprint.pformat(args.__dict__))
    print(model)
    print(pprint.pformat(args.__dict__), file=open(args.results_file, 'a'))
    print(model, file=open(args.results_file, 'a'))

    if args.train:
        train_and_evaluate(model, labeled_dataloader, unlabeled_dataloader, test_dataloader, loss_components_fn, optimizer, args)

    if args.evaluate:
        evaluate(model, test_dataloader, None, args)

    if args.generate:
        generate(model, test_dataloader.dataset, args)

    if args.vis_styles:
        vis_styles(model, args)



