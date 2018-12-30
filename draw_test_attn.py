import os

import torch
from torchvision.datasets import MNIST
import torchvision.transforms as T

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np

from draw import compute_filterbank_matrices


# --------------------
# DRAW paper figure 3
# --------------------

def plot_attn_window(img_tensor, mu_x, mu_y, delta, sigma, attn_window_size, ax):
    ax.imshow(img_tensor.squeeze().data.numpy(), cmap='gray')
    mu_x = mu_x.flatten()
    mu_y = mu_y.flatten()
    x = mu_x[0]
    y = mu_y[0]
    w = mu_y[-1] - mu_y[0]
    h = mu_x[-1] - mu_x[0]
    ax.add_patch(Rectangle((x, y), w, h, facecolor='none', edgecolor='lime', linewidth=5*sigma, alpha=0.7))

def plot_filtered_attn_window(img_tensor, F_x, F_y, g_x, g_y, H, W, attn_window_size, ax):
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.imshow((F_y @ img_tensor @ F_x.permute(0,2,1)).squeeze(), cmap='gray', extent=(g_x - attn_window_size/2,
                                                                                      g_x + attn_window_size/2,
                                                                                      g_y - attn_window_size/2,
                                                                                      g_y + attn_window_size/2))
def test_attn_window_params():
    dataset = MNIST(root='./data', transform=T.ToTensor(), train=True, download=False)
    img = dataset[0][0]
    batch_size, H, W = img.shape

    fig = plt.figure()#figsize=(8,6))
    gs = gridspec.GridSpec(nrows=3, ncols=3, width_ratios=[4,2,1])

    # Figure 3.
    # Left: A 3 × 3 grid of filters superimposed on an image. The stride (δ) and centre location (gX , gY ) are indicated.
    attn_window_size = 3
    g_x = torch.tensor([[-0.2]])
    g_y = torch.tensor([[0.]])
    logvar = torch.tensor([[1.]])
    logdelta = torch.tensor([[-1.]])
    g_x, g_y, delta, mu_x, mu_y, F_x, F_y = compute_filterbank_matrices(g_x, g_y, logvar, logdelta, H, W, attn_window_size)

    ax = fig.add_subplot(gs[:,0])
    ax.imshow(img.squeeze().data.numpy(), cmap='gray')
    ax.scatter(g_x.numpy(), g_y.numpy(), s=150, color='orange', alpha=0.8)
    ax.scatter(mu_x.view(1, -1).repeat(attn_window_size, 1).numpy(),
               mu_y.view(-1,1).repeat(1, attn_window_size).numpy(), s=100, color='lime', alpha=0.8)

    # Right: Three N × N patches extracted from the image (N = 12).
    # The green rectangles on the left indicate the boundary and precision (σ) of the patches, while the patches themselves are shown to the right.
    # The top patch has a small δ and high σ, giving a zoomed-in but blurry view of the centre of the digit;
    # the middle patch has large δ and low σ, effectively downsampling the whole image;
    # and the bottom patch has high δ and σ.
    attn_window_size = 12
    logdeltas = [-1., -0.5, 0.]
    sigmas = [1., 0.5, 3.]

    for i, (logdelta, sigma) in enumerate(zip(logdeltas, sigmas)):
        g_x = torch.tensor([[-0.2]])
        g_y = torch.tensor([[0.]])
        logvar = torch.tensor(sigma**2).float().view(1,-1).log()
        logdelta = torch.tensor(logdelta).float().view(1,-1)

        g_x, g_y, delta, mu_x, mu_y, F_x, F_y = compute_filterbank_matrices(g_x, g_y, logvar, logdelta, H, W, attn_window_size)

        # plot attention window
        ax = fig.add_subplot(gs[i,1])
        plot_attn_window(img, mu_x, mu_y, delta, sigma, attn_window_size, ax)

        # plot attention window zoom in
        ax = fig.add_subplot(gs[i,2])
        plot_filtered_attn_window(img, F_x, F_y, g_x, g_y, H, W, attn_window_size, ax)

    for ax in fig.axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('images/draw_fig_3.png')
    plt.close()


# --------------------
# DRAW paper figure 4 -- Test read write attention on 
# --------------------

def test_read_write_attn():
    #im = cv2.imread('elephant_r.png')
    im = np.asarray(Image.open('images/elephant.png'))
    img = torch.from_numpy(im).float()
    img /= 255.  # normalize to 0-1
    img = img.permute(2,0,1)  # to torch standard (C, H, W)

    print('image dims -- ', img.shape)

    # store dims
    C, H, W = img.shape
    attn_window_size = 12


    # filter params
    g_x = torch.tensor([[0.5]])
    g_y = torch.tensor([[0.5]])
    #logvar = torch.tensor([[1.]]).float().log()
    #logdelta = torch.tensor([[3.]]).float().log()
    logvar = torch.tensor([[1.]])
    logdelta = torch.tensor([[-1.]])

    g_x, g_y, delta, mu_x, mu_y, F_x, F_y = compute_filterbank_matrices(g_x, g_y, logvar, logdelta, H, W, attn_window_size)

    print('delta -- ', delta)
    print('mu_x -- ', mu_x)
    print('mu_y -- ', mu_y)
    print('F_x shape -- ', F_x.shape)

    mu_x = mu_x.flatten()
    mu_y = mu_y.flatten()


    # read image 
    read = F_y @ img @ F_x.transpose(-2,-1)
    print('read image shape -- ', read.shape)
    # reconstruct image
    #read.fill_(1)
    recon_img = 10 * F_y.transpose(-2,-1) @ read @ F_x

    # plot
    fig = plt.figure(figsize=(9, 3))
    gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[W,attn_window_size,W])


    # show original image with attention bbox
    ax = fig.add_subplot(gs[0,0])
    ax.imshow(im)
    x = mu_x[0]
    y = mu_y[0]
    w = mu_y[-1] - mu_y[0]
    h = mu_x[-1] - mu_x[0]
    ax.add_patch(Rectangle((x, y), w, h, facecolor='none', edgecolor='lime', linewidth=5, alpha=0.7))

    # show attention patch
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, attn_window_size)
    ax.set_ylim(0, H)
    ax.imshow(read.squeeze().data.numpy().transpose(1,2,0), extent = (0, attn_window_size, H/2 - attn_window_size/2, H/2 + attn_window_size/2))

    # show reconstruction
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(recon_img.squeeze().data.numpy().transpose(1,2,0))

    plt.tight_layout()
    for ax in plt.gcf().axes:
        ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
    plt.savefig('images/draw_fig_4.png')
    plt.close()



if __name__ == '__main__':
    test_attn_window_params()
    test_read_write_attn()
