import os
import sys
import shutil
import json
from datetime import datetime
import torch

from tensorboardX import SummaryWriter

from subprocess import check_call


def set_writer(log_path, comment='', restore=False):
    """ setup a tensorboardx summarywriter """
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if restore:
        log_path = os.path.dirname(log_path)
    else:
        log_path = os.path.join(log_path, current_time + comment)
    writer = SummaryWriter(log_dir=log_path)
    return writer


def save_checkpoint(state, checkpoint, is_best=None, quiet=False):
    """ saves model and training params at checkpoint + 'last.pt'; if is_best also saves checkpoint + 'best.pt'

    args
        state -- dict; with keys model_state_dict, optimizer_state_dict, epoch, scheduler_state_dict, etc
        is_best -- bool; true if best model seen so far
        checkpoint -- str; folder where params are to be saved
    """

    filepath = os.path.join(checkpoint, 'state_checkpoint.pt')
    if not os.path.exists(checkpoint):
        if not quiet:
            print('Checkpoint directory does not exist Making directory {}'.format(checkpoint))
        os.mkdir(checkpoint)

    torch.save(state, filepath)

#    if is_best:
#        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_state_checkpoint.pt'))

    if not quiet:
        print('Checkpoint saved.')


def load_checkpoint(checkpoint, models, optimizers=None, scheduler=None, best_metric=None, map_location='cpu'):
    """ loads model state_dict from filepath; if optimizer and lr_scheduler provided also loads them

    args
        checkpoint -- string of filename
        model -- torch nn.Module model
        optimizer -- torch.optim instance to resume from checkpoint
        lr_scheduler -- torch.optim.lr_scheduler instance to resume from checkpoint
    """

    if not os.path.exists(checkpoint):
        raise('File does not exist {}'.format(checkpoint))

    checkpoint = torch.load(checkpoint, map_location=map_location)
    models = [m.load_state_dict(checkpoint['model_state_dicts'][i]) for i, m in enumerate(models)]

    if optimizers:
        try:
            optimizers = [o.load_state_dict(checkpoint['optimizer_state_dicts'][i]) for i, o in enumerate(optimizers)]
        except KeyError:
            print('No optimizer state dict in checkpoint file')

    if best_metric:
        try:
            best_metric = checkpoint['best_val_acc']
        except KeyError:
            print('No best validation accuracy recorded in checkpoint file.')

    if scheduler:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except KeyError:
            print('No lr scheduler state dict in checkpoint file')

    return checkpoint['epoch']

