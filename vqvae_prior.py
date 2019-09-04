"""
Implementation of VQ-VAE-2 priors:
    -- van den Oord, 'Generating Diverse High-Fidelity Images with VQ-VAE-2' -- https://arxiv.org/abs/1906.00446
    -- van den Oord, 'Conditional Image Generation with PixelCNN Decoders' -- https://arxiv.org/abs/1606.05328
    -- Xi Chen, 'PixelSNAIL: An Improved Autoregressive Generative Model' -- https://arxiv.org/abs/1712.09763
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image, make_grid

import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

import os
import argparse
import time
import pprint
from functools import partial

from vqvae import VQVAE2, fetch_vqvae_dataloader, load_model, save_json, load_json
from optim import Adam, RMSprop


parser = argparse.ArgumentParser()

# action
parser.add_argument('--train', action='store_true', help='Train model.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate model.')
parser.add_argument('--generate', action='store_true', help='Generate samples from a model.')
parser.add_argument('--seed', type=int, default=0, help='Random seed to use.')
parser.add_argument('--cuda', type=int, help='Which cuda device to use.')
parser.add_argument('--mini_data', action='store_true', help='Truncate dataset to a single minibatch.')
# model
parser.add_argument('--which_prior', choices=['bottom', 'top'], help='Which prior model to train.')
parser.add_argument('--vqvae_dir', type=str, required=True, help='Path to VQVAE folder with config.json and checkpoint.pt files.')
parser.add_argument('--n_channels', default=128, type=int, help='Number of channels for gated residual convolutional blocks.')
parser.add_argument('--n_out_conv_channels', default=1024, type=int, help='Number of channels for outer 1x1 convolutional layers.')
parser.add_argument('--n_res_layers', default=20, type=int, help='Number of Gated Residual Blocks.')
parser.add_argument('--n_cond_classes', default=5, type=int, help='Number of classes if conditional model.')
parser.add_argument('--n_cond_stack_layers', default=10, type=int, help='Number of conditioning stack residual blocks.')
parser.add_argument('--n_out_stack_layers', default=10, type=int, help='Number of output stack layers.')
parser.add_argument('--kernel_size', default=5, type=int, help='Kernel size for the gated residual convolutional blocks.')
parser.add_argument('--drop_rate', default=0, type=float, help='Dropout for the Gated Residual Blocks.')
# data params
parser.add_argument('--output_dir', type=str, help='Location where weights, logs, and sample should be saved.')
parser.add_argument('--restore_dir', nargs='+', help='Location where configs and weights are to be restored from.')
parser.add_argument('--n_bits', type=int, help='Number of bits of input data.')
# training param
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
parser.add_argument('--lr_decay', type=float, default=0.99999, help='Learning rate decay (assume 5e-5 @ 300k iters for lr 0.001).')
parser.add_argument('--polyak', type=float, default=0.9995, help='Polyak decay for exponential moving averaging.')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--step', type=int, default=0, help='Current step of training (number of minibatches processed).')
parser.add_argument('--start_epoch', default=0, help='Starting epoch (for logging; to be overwritten when restoring file.')
parser.add_argument('--log_interval', type=int, default=50, help='How often to show loss statistics and save samples.')
parser.add_argument('--eval_interval', type=int, default=10, help='How often to evaluate and save samples.')
parser.add_argument('--save_interval', type=int, default=300, help='How often to evaluate and save samples.')
# distributed training params
parser.add_argument('--distributed', action='store_true', default=False, help='Whether to use DistributedDataParallels on multiple machines and GPUs.')
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--rank', type=int, default=0)
# generation param
parser.add_argument('--n_samples', type=int, default=8, help='Number of samples to generate.')



# --------------------
# Data and model loading
# --------------------

@torch.no_grad()
def extract_codes_from_dataloader(vqvae, dataloader, dataset_path):
    """ encode image inputs with vqvae and extract field of discrete latents (the embedding indices in the codebook with closest l2 distance) """
    device = next(vqvae.parameters()).device
    e1s, e2s, ys = [], [], []
    for x, y in tqdm(dataloader):
        z_e = vqvae.encode(x.to(device))
        encoding_indices, _ = vqvae.quantize(z_e)  # tuple of (bottom, top encoding indices) where each is (B,1,H,W)

        e1, e2 = encoding_indices
        e1s.append(e1)
        e2s.append(e2)
        ys.append(y)
    return TensorDataset(torch.cat(e1s).cpu(), torch.cat(e2s).cpu(), torch.cat(ys))

def maybe_extract_codes(vqvae, args, train):
    """ construct datasets of vqvae encodings and class conditional labels -- each dataset entry is [encodings level 1 (bottom), encodings level 2 (top), class label vector] """
    # paths to load/save as `chexpert_train_codes_mini_data.pt`
    dataset_path = os.path.join(args.vqvae_dir, '{}_{}_codes'.format(args.dataset, 'train' if train else 'valid') + args.mini_data*'_mini_data_{}'.format(args.batch_size) + '.pt')
    if not os.path.exists(dataset_path):
        print('Extracting codes for {} data ...'.format('train' if train else 'valid'))
        dataloader = fetch_vqvae_dataloader(args, train)
        dataset = extract_codes_from_dataloader(vqvae, dataloader, dataset_path)
        torch.save(dataset, dataset_path)
    else:
        dataset = torch.load(dataset_path)
    if args.on_main_process: print('Loaded {} codes dataset of size {}'.format('train' if train else 'valid', len(dataset)))
    return dataset

def fetch_prior_dataloader(vqvae, args, train=True):
    dataset = maybe_extract_codes(vqvae, args, train)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.distributed and train else None
    return DataLoader(dataset, args.batch_size, shuffle=(train and sampler is None), sampler=sampler, num_workers=4, pin_memory=('cuda' in args.device))

def preprocess(x, n_bits):
    """ preprosses discrete latents space [0, 2**n_bits) to model space [-1,1]; if size of the codebook ie n_embeddings = 512 = 2**9 -> n_bit=9 """
    # 1. convert data to float
    # 2. normalize to [0,1] given quantization
    # 3. shift to [-1,1]
    return x.float().div(2**n_bits - 1).mul(2).add(-1)

def deprocess(x, n_bits):
    """ deprocess x from model space [-1,1] to discrete latents space [0, 2**n_bits) where 2**n_bits is size of the codebook """
    # 1. shift to [0,1]
    # 2. quantize to n_bits
    # 3. convert data to long
    return x.add(1).div(2).mul(2**n_bits - 1).long()

# --------------------
# PixelSNAIL -- top level prior conditioned on class labels
# --------------------

def down_shift(x):
    return F.pad(x, (0,0,1,0))[:,:,:-1,:]

def right_shift(x):
    return F.pad(x, (1,0))[:,:,:,:-1]

def concat_elu(x):
    return F.elu(torch.cat([x, -x], dim=1))

class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.utils.weight_norm(self)

class DownShiftedConv2d(Conv2d):
    def forward(self, x):
        # pad H above and W on each side
        Hk, Wk = self.kernel_size
        x = F.pad(x, ((Wk-1)//2, (Wk-1)//2, Hk-1, 0))
        return super().forward(x)

class DownRightShiftedConv2d(Conv2d):
    def forward(self, x):
        # pad above and on left (ie shift input down and right)
        Hk, Wk = self.kernel_size
        x = F.pad(x, (Wk-1, 0, Hk-1, 0))
        return super().forward(x)

class GatedResidualLayer(nn.Module):
    def __init__(self, conv, n_channels, kernel_size, drop_rate=0, shortcut_channels=None, n_cond_classes=None, relu_fn=concat_elu):
        super().__init__()
        self.relu_fn = relu_fn

        self.c1 = conv(2*n_channels, n_channels, kernel_size)
        if shortcut_channels:
            self.c1c = Conv2d(2*shortcut_channels, n_channels, kernel_size=1)
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        self.c2 = conv(2*n_channels, 2*n_channels, kernel_size)
        if n_cond_classes:
            self.proj_y = nn.Linear(n_cond_classes, 2*n_channels)

    def forward(self, x, a=None, y=None):
        c1 = self.c1(self.relu_fn(x))
        if a is not None:  # shortcut connection if auxiliary input 'a' is given
            c1 = c1 + self.c1c(self.relu_fn(a))
        c1 = self.relu_fn(c1)
        if hasattr(self, 'dropout'):
            c1 = self.dropout(c1)
        c2 = self.c2(c1)
        if y is not None:
            c2 += self.proj_y(y)[:,:,None,None]
        a, b = c2.chunk(2,1)
        out = x + a * torch.sigmoid(b)
        return out

def causal_attention(k, q, v, mask, nh, drop_rate, training):
    B, dq, H, W = q.shape
    _, dv, _, _ = v.shape

    # split channels into multiple heads, flatten H,W dims and scale q; out (B, nh, dkh or dvh, HW)
    flat_q = q.reshape(B, nh, dq//nh, H, W).flatten(3) * (dq//nh)**-0.5
    flat_k = k.reshape(B, nh, dq//nh, H, W).flatten(3)
    flat_v = v.reshape(B, nh, dv//nh, H, W).flatten(3)

    logits = torch.matmul(flat_q.transpose(2,3), flat_k)              # (B,nh,HW,dq) dot (B,nh,dq,HW) = (B,nh,HW,HW)
    logits = F.dropout(logits, p=drop_rate, training=training, inplace=True)
    logits = logits.masked_fill(mask==0, -1e10)
    weights = F.softmax(logits, -1)

    attn_out = torch.matmul(weights, flat_v.transpose(2,3))           # (B,nh,HW,HW) dot (B,nh,HW,dvh) = (B,nh,HW,dvh)
    attn_out = attn_out.transpose(2,3)                                # (B,nh,dvh,HW)
    return attn_out.reshape(B, -1, H, W)                              # (B,dv,H,W)

class AttentionGatedResidualLayer(nn.Module):
    def __init__(self, n_channels, n_background_ch, n_res_layers, n_cond_classes, drop_rate, nh, dq, dv, attn_drop_rate):
        super().__init__()
        # attn params
        self.nh = nh
        self.dq = dq
        self.dv = dv
        self.attn_drop_rate = attn_drop_rate

        self.input_gated_resnet = nn.ModuleList([
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)]])
        self.in_proj_kv = nn.Sequential(GatedResidualLayer(Conv2d, 2*n_channels + n_background_ch, 1, drop_rate, None, n_cond_classes),
                                        Conv2d(2*n_channels + n_background_ch, dq+dv, 1))
        self.in_proj_q  = nn.Sequential(GatedResidualLayer(Conv2d, n_channels + n_background_ch, 1, drop_rate, None, n_cond_classes),
                                        Conv2d(n_channels + n_background_ch, dq, 1))
        self.out_proj = GatedResidualLayer(Conv2d, n_channels, 1, drop_rate, dv, n_cond_classes)

    def forward(self, x, background, attn_mask, y=None):
        ul = x
        for m in self.input_gated_resnet:
            ul = m(ul, y=y)

        kv = self.in_proj_kv(torch.cat([x, ul, background], 1))
        k, v = kv.split([self.dq, self.dv], 1)
        q = self.in_proj_q(torch.cat([ul, background], 1))
        attn_out = causal_attention(k, q, v, attn_mask, self.nh, self.attn_drop_rate, self.training)
        return self.out_proj(ul, attn_out)

class PixelSNAIL(nn.Module):
    def __init__(self, input_dims, n_channels, n_res_layers, n_out_stack_layers, n_cond_classes, n_bits,
                 attn_n_layers=4, attn_nh=8, attn_dq=16, attn_dv=128, attn_drop_rate=0, drop_rate=0.5, **kwargs):
        super().__init__()
        H,W = input_dims[2]
        # init background
        background_v = ((torch.arange(H, dtype=torch.float) - H / 2) / 2).view(1,1,-1,1).expand(1,1,H,W)
        background_h = ((torch.arange(W, dtype=torch.float) - W / 2) / 2).view(1,1,1,-1).expand(1,1,H,W)
        self.register_buffer('background', torch.cat([background_v, background_h], 1))
        # init attention mask over current and future pixels
        attn_mask = torch.tril(torch.ones(1,1,H*W,H*W), diagonal=-1).byte()  # 1s below diagonal -- attend to context only
        self.register_buffer('attn_mask', attn_mask)

        # input layers for `up` and `up and to the left` pixels
        self.ul_input_d = DownShiftedConv2d(2, n_channels, kernel_size=(1,3))
        self.ul_input_dr = DownRightShiftedConv2d(2, n_channels, kernel_size=(2,1))
        self.ul_modules = nn.ModuleList([
            *[AttentionGatedResidualLayer(n_channels, self.background.shape[1], n_res_layers, n_cond_classes, drop_rate,
                                          attn_nh, attn_dq, attn_dv, attn_drop_rate) for _ in range(attn_n_layers)]])
        self.output_stack = nn.Sequential(
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, None, n_cond_classes) \
            for _ in range(n_out_stack_layers)])
        self.output_conv = Conv2d(n_channels, 2**n_bits, kernel_size=1)


    def forward(self, x, y=None):
        # add channel of ones to distinguish image from padding later on
        x = F.pad(x, (0,0,0,0,0,1), value=1)

        ul = down_shift(self.ul_input_d(x)) + right_shift(self.ul_input_dr(x))
        for m in self.ul_modules:
            ul = m(ul, self.background.expand(x.shape[0],-1,-1,-1), self.attn_mask, y)
        ul = self.output_stack(ul)
        return self.output_conv(F.elu(ul)).unsqueeze(2)  # out (B, 2**n_bits, 1, H, W)

# --------------------
# PixelCNN -- bottom level prior conditioned on class labels and top level codes
# --------------------

def pixelcnn_gate(x):
    a, b = x.chunk(2,1)
    return torch.tanh(a) * torch.sigmoid(b)

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        self.mask_type = mask_type
        super().__init__(*args, **kwargs)

    def apply_mask(self):
        H, W = self.kernel_size
        self.weight.data[:,:,H//2+1:,:].zero_()     # mask out rows below the middle
        self.weight.data[:,:,H//2,W//2+1:].zero_()  # mask out center row pixels right of middle
        if self.mask_type=='a':
            self.weight.data[:,:,H//2,W//2] = 0     # mask out center pixel

    def forward(self, x):
        self.apply_mask()
        return super().forward(x)

class GatedResidualBlock(nn.Module):
    """ Figure 2 in Conditional image generation with PixelCNN Decoders """
    def __init__(self, in_channels, out_channels, kernel_size, n_cond_channels, drop_rate):
        super().__init__()
        self.residual = (in_channels==out_channels)
        self.drop_rate = drop_rate

        self.v   = nn.Conv2d(in_channels, 2*out_channels, kernel_size, padding=kernel_size//2)              # vertical stack
        self.h   = nn.Conv2d(in_channels, 2*out_channels, (1, kernel_size), padding=(0, kernel_size//2))    # horizontal stack
        self.v2h = nn.Conv2d(2*out_channels, 2*out_channels, kernel_size=1)                                 # vertical to horizontal connection
        self.h2h = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)                         # horizontal to horizontal

        if n_cond_channels:
            self.in_proj_y = nn.Conv2d(n_cond_channels, 2*out_channels, kernel_size=1)

        if self.drop_rate > 0:
            self.dropout_h = nn.Dropout(drop_rate)

    def apply_mask(self):
        self.v.weight.data[:,:,self.v.kernel_size[0]//2:,:].zero_()     # mask out middle row and below
        self.h.weight.data[:,:,:,self.h.kernel_size[1]//2+1:].zero_()   # mask out to the right of the central column

    def forward(self, x_v, x_h, y):
        self.apply_mask()

        # projection of y if included for conditional generation (cf paper section 2.3 -- added before the pixelcnn_gate)
        proj_y = self.in_proj_y(y)

        # vertical stack
        x_v_out = self.v(x_v)
        x_v2h = self.v2h(x_v_out) + proj_y
        x_v_out = pixelcnn_gate(x_v_out)

        # horizontal stack
        x_h_out = self.h(x_h) + x_v2h + proj_y
        x_h_out = pixelcnn_gate(x_h_out)
        if self.drop_rate:
            x_h_out = self.dropout_h(x_h_out)
        x_h_out = self.h2h(x_h_out)

        # residual connection
        if self.residual:
            x_h_out = x_h_out + x_h

        return x_v_out, x_h_out

    def extra_repr(self):
        return 'residual={}, drop_rate={}'.format(self.residual, self.drop_rate)

class PixelCNN(nn.Module):
    def __init__(self, n_channels, n_out_conv_channels, kernel_size, n_res_layers, n_cond_stack_layers, n_cond_classes, n_bits,
                 drop_rate=0, **kwargs):
        super().__init__()
        # conditioning layers (bottom prior conditioned on class labels and top-level code)
        self.in_proj_y = nn.Linear(n_cond_classes, 2*n_channels)
        self.in_proj_h = nn.ConvTranspose2d(1, n_channels, kernel_size=4, stride=2, padding=1)  # upsample top codes to bottom-level spacial dim
        self.cond_layers = nn.ModuleList([
            GatedResidualLayer(partial(Conv2d, padding=kernel_size//2), n_channels, kernel_size, drop_rate, None, n_cond_classes) \
            for _ in range(n_cond_stack_layers)])
        self.out_proj_h = nn.Conv2d(n_channels, 2*n_channels, kernel_size=1)  # double channels top apply pixelcnn_gate

        # pixelcnn layers
        self.input_conv = MaskedConv2d('a', 1, 2*n_channels, kernel_size=7, padding=3)
        self.res_layers = nn.ModuleList([
            GatedResidualBlock(n_channels, n_channels, kernel_size, 2*n_channels, drop_rate) for _ in range(n_res_layers)])
        self.conv_out1 = nn.Conv2d(n_channels, 2*n_out_conv_channels, kernel_size=1)
        self.conv_out2 = nn.Conv2d(n_out_conv_channels, 2*n_out_conv_channels, kernel_size=1)
        self.output = nn.Conv2d(n_out_conv_channels, 2**n_bits, kernel_size=1)

    def forward(self, x, h=None, y=None):
        # conditioning inputs -- h is top-level codes; y is class labels
        h = self.in_proj_h(h)
        for l in self.cond_layers:
            h = l(h, y=y)
        h = self.out_proj_h(h)
        y = self.in_proj_y(y)[:,:,None,None]

        # pixelcnn model
        x = pixelcnn_gate(self.input_conv(x) + h + y)
        x_v, x_h = x, x
        for l in self.res_layers:
            x_v, x_h = l(x_v, x_h, y)
        out = pixelcnn_gate(self.conv_out1(x_h))
        out = pixelcnn_gate(self.conv_out2(out))
        return self.output(out).unsqueeze(2)  # (B, 2**n_bits, 1, H, W)

# --------------------
# Train and evaluate
# --------------------

def train_epoch(model, dataloader, optimizer, scheduler, epoch, writer, args):
    model.train()

    tic = time.time()
    if args.on_main_process: pbar = tqdm(total=len(dataloader), desc='epoch {}/{}'.format(epoch, args.start_epoch + args.n_epochs))
    for e1, e2, y in dataloader:
        args.step += args.world_size

        e1, e2, y = e1.to(args.device), e2.to(args.device), y.to(args.device)

        if args.which_prior == 'bottom':
            x = e1
            logits = model(preprocess(x, args.n_bits), preprocess(e2, args.n_bits), y)
        elif args.which_prior == 'top':
            x = e2
            logits = model(preprocess(x, args.n_bits), y)
        loss = F.cross_entropy(logits, x).mean(0)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()
        if scheduler: scheduler.step()

        # record
        if args.on_main_process:
            pbar.set_postfix(loss='{:.4f}'.format(loss.item() / np.log(2)))
            pbar.update()

        if args.step % args.log_interval == 0 and args.on_main_process:
            writer.add_scalar('train_bits_per_dim', loss.item() / np.log(2), args.step)

        # save
        if args.step % args.save_interval == 0 and args.on_main_process:
            # save model
            torch.save({'epoch': epoch,
                        'global_step': args.step,
                        'state_dict': model.module.state_dict() if args.distributed else model.state_dict()},
                        os.path.join(args.output_dir, 'checkpoint.pt'))
            torch.save(optimizer.state_dict(), os.path.join(args.output_dir, 'optim_checkpoint.pt'))
            if scheduler: torch.save(scheduler.state_dict(), os.path.join(args.output_dir, 'sched_checkpoint.pt'))

    if args.on_main_process: pbar.close()

@torch.no_grad()
def evaluate(model, dataloader, args):
    model.eval()

    losses = 0
    for e1, e2, y in dataloader:
        e1, e2, y = e1.to(args.device), e2.to(args.device), y.to(args.device)
        if args.which_prior == 'bottom':
            x = e1
            logits = model(preprocess(x, args.n_bits), preprocess(e2, args.n_bits), y)
        elif args.which_prior == 'top':
            x = e2
            logits = model(preprocess(x, args.n_bits), y)
        losses += F.cross_entropy(logits, x).mean(0).item()
    return losses / (len(dataloader) * np.log(2))  # to bits per dim

def train_and_evaluate(model, vqvae, train_dataloader, valid_dataloader, optimizer, scheduler, writer, args):
    for epoch in range(args.start_epoch, args.start_epoch + args.n_epochs):
        train_epoch(model, train_dataloader, optimizer, scheduler, epoch, writer, args)

        if (epoch+1) % args.eval_interval == 0:
#            optimizer.use_ema(True)

            # evaluate
            eval_bpd = evaluate(model, valid_dataloader, args)
            if args.on_main_process:
                print('Evaluate bits per dim: {:.4f}'.format(eval_bpd))
                writer.add_scalar('eval_bits_per_dim', eval_bpd, args.step)

            # generate
            samples = generate_samples_in_training(model, vqvae, train_dataloader, args)
            samples = make_grid(samples, normalize=True, nrow=args.n_samples)
            if args.distributed:
                # collect samples tensor from all processes onto main process cpu
                tensors = [torch.empty(samples.shape, dtype=samples.dtype).cuda() for i in range(args.world_size)]
                torch.distributed.all_gather(tensors, samples)
                samples = torch.cat(tensors, 2)
            if args.on_main_process:
                samples = samples.cpu()
                writer.add_image('samples_' + args.which_prior, samples, args.step)
                save_image(samples, os.path.join(args.output_dir, 'samples_{}_step_{}.png'.format(args.which_prior, args.step)))

#            optimizer.use_ema(False)

        if args.on_main_process:
            # save model
            torch.save({'epoch': epoch,
                        'global_step': args.step,
                        'state_dict': model.module.state_dict() if args.distributed else model.state_dict()},
                        os.path.join(args.output_dir, 'checkpoint.pt'))
            torch.save(optimizer.state_dict(), os.path.join(args.output_dir, 'optim_checkpoint.pt'))
            if scheduler: torch.save(scheduler.state_dict(), os.path.join(args.output_dir, 'sched_checkpoint.pt'))


# --------------------
# Sample and generate
# --------------------

def sample_prior(model, h, y, n_samples, input_dims, n_bits):
    model.eval()

    H,W = input_dims
    out = torch.zeros(n_samples, 1, H, W, device=next(model.parameters()).device)
    if args.on_main_process: pbar = tqdm(total=H*W, desc='Generating {} images'.format(n_samples))
    for hi in range(H):
        for wi in range(W):
            logits = model(out, y) if h is None else model(out, h, y)
            probs = F.softmax(logits, dim=1)
            sample = torch.multinomial(probs[:,:,:,hi,wi].squeeze(2), 1)
            out[:,:,hi,wi] = preprocess(sample, n_bits)  # multinomial samples long tensor in [0, 2**n_bits), convert back to model space [-1,1]
            if args.on_main_process: pbar.update()
            del logits, probs, sample
    if args.on_main_process: pbar.close()
    return deprocess(out, n_bits)  # out (B,1,H,W) field of latents in latent space [0, 2**n_bits)

@torch.no_grad()
def generate(vqvae, bottom_model, top_model, args, ys=None):
    samples = []
    for y in ys.unsqueeze(1):  # condition on class one-hot labels; (n_samples, 1, n_cond_classes) when sliced on dim 0 returns (1,n_cond_classes)
        # sample top prior conditioned on class labels y
        top_samples = sample_prior(top_model, None, y, args.n_samples, args.input_dims[2], args.n_bits)
        # sample bottom prior conditioned on top_sample codes and class labels y
        bottom_samples = sample_prior(bottom_model, preprocess(top_samples, args.n_bits), y, args.n_samples, args.input_dims[1], args.n_bits)
        # decode
        samples += [vqvae.decode(None, vqvae.embed((bottom_samples, top_samples)))]
    samples = torch.cat(samples)
    return make_grid(samples, normalize=True, scale_each=True)

def generate_samples_in_training(model, vqvae, dataloader, args):
    if args.which_prior == 'top':
        # zero out bottom samples so no contribution
        bottom_samples = torch.zeros(args.n_samples*(args.n_cond_classes+1),1,*args.input_dims[1], dtype=torch.long)
        # sample top prior
        top_samples = []
        for y in torch.eye(args.n_cond_classes + 1, args.n_cond_classes).unsqueeze(1).to(args.device):  # note eg: torch.eye(3,2) = [[1,0],[0,1],[0,0]]
            top_samples += [sample_prior(model, None, y, args.n_samples, args.input_dims[2], args.n_bits).cpu()]
        top_samples = torch.cat(top_samples)
        # decode
        samples = vqvae.decode(z_e=None, z_q=vqvae.embed((bottom_samples.to(args.device), top_samples.to(args.device))))

    elif args.which_prior == 'bottom':  # level 1
        # use the dataset ground truth top codes and only sample the bottom
        bottom_gt, top_gt, y = next(iter(dataloader))  # take e2 and y from dataloader output (e1,e2,y)
        bottom_gt, top_gt, y = bottom_gt[:args.n_samples].to(args.device), top_gt[:args.n_samples].to(args.device), y[:args.n_samples].to(args.device)
        # sample bottom prior
        bottom_samples = sample_prior(model, preprocess(top_gt, args.n_bits), y, args.n_samples, args.input_dims[1], args.n_bits)
        # decode
        # stack (1) recon using bottom+top actual latents,
        #       (2) recon using top latents only,
        #       (3) recon using top latent and bottom prior samples
        recon_actuals = vqvae.decode(z_e=None, z_q=vqvae.embed((bottom_gt, top_gt)))
        recon_top = vqvae.decode(z_e=None, z_q=vqvae.embed((bottom_gt.fill_(0), top_gt)))
        recon_samples = vqvae.decode(z_e=None, z_q=vqvae.embed((bottom_samples, top_gt)))
        samples = torch.cat([recon_actuals, recon_top, recon_samples])

    return samples


# --------------------
# Main
# --------------------

if __name__ == '__main__':
    args = parser.parse_args()
    if args.restore_dir and args.which_prior:
        args.output_dir = args.restore_dir[0]
    if not args.output_dir:  # if not given or not set by restore_dir use results/file_name/time_stamp
        # name experiment 'vqvae_[vqvae_dir]_prior_[prior_args]_[timestamp]'
        exp_name = 'vqvae_' + args.vqvae_dir.strip('/').rpartition('/')[2] + \
                    '_prior_{which_prior}' + args.mini_data*'_mini{}'.format(args.batch_size) + \
                    '_b{batch_size}_c{n_channels}_outc{n_out_conv_channels}_nres{n_res_layers}_condstack{n_cond_stack_layers}' + \
                    '_outstack{n_out_stack_layers}_drop{drop_rate}' + \
                    '_{}'.format(time.strftime('%Y-%m-%d_%H-%M', time.gmtime()))
        args.output_dir = './results/{}/{}'.format(os.path.splitext(__file__)[0], exp_name.format(**args.__dict__))
        os.makedirs(args.output_dir, exist_ok=True)

    # setup device and distributed training
    if args.distributed:
        args.cuda = int(os.environ['LOCAL_RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(args.cuda)
        args.device = 'cuda:{}'.format(args.cuda)

        # initialize
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        args.device = 'cuda:{}'.format(args.cuda) if args.cuda is not None and torch.cuda.is_available() else 'cpu'

    # write ops only when on_main_process
    args.on_main_process = (args.distributed and args.cuda == 0) or not args.distributed

    # setup seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load vqvae
    #   load config; extract bits and input sizes throughout the hierarchy from the vqvae config
    vqvae_config = load_json(os.path.join(args.vqvae_dir, 'config.json'))
    img_dims = vqvae_config['input_dims'][1:]
    args.input_dims = [img_dims, [img_dims[0]//4, img_dims[1]//4], [img_dims[0]//8, img_dims[1]//8]]
    args.n_bits = int(np.log2(vqvae_config['n_embeddings']))
    args.dataset = vqvae_config['dataset']
    args.data_dir = vqvae_config['data_dir']
    #   load model
    vqvae, _, _ = load_model(VQVAE2, vqvae_config, args.vqvae_dir, args, restore=True, eval_mode=True, verbose=args.on_main_process)
    #   reset start_epoch and step after model loading
    args.start_epoch, args.step = 0, 0
    #   expose functions
    if args.distributed:
        vqvae.encode = vqvae.module.encode
        vqvae.decode = vqvae.module.decode
        vqvae.embed  = vqvae.module.embed


    # load prior model
    #   save prior config to feed to load_model
    if not os.path.exists(os.path.join(args.output_dir, 'config_{}.json'.format(args.cuda))):
        save_json(args.__dict__, 'config_{}'.format(args.cuda), args)
    #   load model + optimizers, scheduler if training
    if args.which_prior:
        model, optimizer, scheduler = load_model(PixelCNN if args.which_prior=='bottom' else PixelSNAIL,
                                                 config=args.__dict__,
                                                 model_dir=args.output_dir,
                                                 args=args,
                                                 restore=(args.restore_dir is not None),
                                                 eval_mode=False,
                                                 optimizer_cls=partial(RMSprop,
                                                                       lr=args.lr,
                                                                       polyak=args.polyak),
                                                 scheduler_cls=partial(torch.optim.lr_scheduler.ExponentialLR, gamma=args.lr_decay),
                                                 verbose=args.on_main_process)
    else:
        assert args.restore_dir and len(args.restore_dir)==2, '`restore_dir` should specify restore dir to bottom prior and top prior'
        # load both top and bottom to generate
        restore_bottom, restore_top = args.restore_dir
        bottom_model, _, _ = load_model(PixelCNN, config=None, model_dir=restore_bottom, args=args, restore=True, eval_mode=True,
                                        optimizer_cls=partial(RMSprop, lr=args.lr, polyak=args.polyak))
        top_model, _, _    = load_model(PixelSNAIL, config=None, model_dir=restore_top, args=args, restore=True, eval_mode=True,
                                        optimizer_cls=partial(RMSprop, lr=args.lr, polyak=args.polyak))

    # save and print config and setup writer on main process
    writer = None
    if args.on_main_process:
        pprint.pprint(args.__dict__)
        writer = SummaryWriter(log_dir = args.output_dir)
        writer.add_text('config', str(args.__dict__))

    if args.train:
        assert args.which_prior is not None, 'Must specify `which_prior` to train.'
        train_dataloader = fetch_prior_dataloader(vqvae, args, True)
        valid_dataloader = fetch_prior_dataloader(vqvae, args, False)
        train_and_evaluate(model, vqvae, train_dataloader, valid_dataloader, optimizer, scheduler, writer, args)

    if args.evaluate:
        assert args.which_prior is not None, 'Must specify `which_prior` to evaluate.'
        valid_dataloader = fetch_prior_dataloader(vqvae, args, False)
#        optimizer.use_ema(True)
        eval_bpd = evaluate(model, valid_dataloader, args)
        if args.on_main_process:
            print('Evaluate bits per dim: {:.4f}'.format(eval_bpd))

    if args.generate:
        assert args.which_prior is None, 'Remove `which_prior` to load both priors and generate'
#        optimizer.use_ema(True)
        samples = generate(vqvae, bottom_model, top_model, args, ys=torch.eye(args.n_cond_classes + 1, args.n_cond_classes).to(args.device))
        if args.distributed:
            torch.manual_seed(args.rank)
            # collect samples tensor from all processes onto main process cpu
            tensors = [torch.empty(samples.shape, dtype=samples.dtype).cuda() for i in range(args.world_size)]
            torch.distributed.all_gather(tensors, samples)  # collect samples tensor from all processes onto main process cpu
            samples = torch.cat(tensors, 2)
        if args.on_main_process:
            samples = samples.cpu()
            writer.add_image('samples', samples, args.step)
            save_image(samples.cpu(), os.path.join(args.output_dir, 'generation_sample_step_{}.png'.format(args.step)))

