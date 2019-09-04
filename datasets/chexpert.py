import os
import math
import time
import glob
from multiprocessing import Pool
from functools import partial

import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class ChexpertDataset(Dataset):
    attr_all_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                      'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                      'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                      'Fracture', 'Support Devices']
    # subset of labels to use
    attr_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    def __init__(self, root, train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.input_dims = (1, *(int(i) for i in self.root.strip('/').rpartition('/')[2].split('_')[-1].split('x')))

        if train:
            self.data = self._load_and_preprocess_training_data(os.path.join(self.root, 'train.csv'))
        else:
            self.data = pd.read_csv(os.path.join(self.root, 'valid.csv'), keep_default_na=True)

        # store index of the selected attributes in the columns of the data for faster indexing
        self.attr_idxs = [self.data.columns.tolist().index(a) for a in self.attr_names]

    def __getitem__(self, idx):
        # 1. select and load image
        img_path = self.data.iloc[idx, 1]  # `Path` is the first column after index
        img = Image.open(os.path.join(self.root, img_path.partition('/')[2]))
        if self.transform is not None:
            img = self.transform(img)

        # 2. select attributes as targets
        attr = self.data.iloc[idx, self.attr_idxs].values.astype(float)
        attr = torch.from_numpy(attr).float()

        return img, attr

    def __len__(self):
        return len(self.data)

    def _load_and_preprocess_training_data(self, csv_path):
        # Dataset labels are: blank for unmentioned, 0 for negative, -1 for uncertain, and 1 for positive.
        # Process by:
        #    1. fill NAs (blanks for unmentioned) as 0 (negatives)
        #    2. fill -1 as 1 (U-Ones method described in paper)

        # load
        train_df = pd.read_csv(csv_path, keep_default_na=True)

        # 1. fill NAs (blanks for unmentioned) as 0 (negatives)
        # attr columns ['No Finding', ..., 'Support Devices']; note AP/PA remains with NAs for Lateral pictures
        train_df[self.attr_names] = train_df[self.attr_names].fillna(0)

        # 2. fill -1 as 1 (U-Ones method described in paper)
        train_df[self.attr_names] = train_df[self.attr_names].replace(-1,1)

        return train_df


def compute_dataset_mean_and_std(dataset):
    m = 0
    s = 0
    k = 1
    for img, _, _ in tqdm(dataset):
        x = img.mean().item()
        new_m = m + (x - m)/k
        s += (x - m)*(x - new_m)
        m = new_m
        k += 1
    print('Number of datapoints: ', k)
    return m, math.sqrt(s/(k-1))

# --------------------
# Resize dataset
# --------------------

def _process_entry(img_path, root, source_dir, target_dir, transforms):
    # img_path is e.g. `CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg`
    subpath = img_path.partition('/')[-1]

    # make new img folders/subfolders
    os.makedirs(os.path.dirname(os.path.join(root, target_dir, subpath)), exist_ok=True)

    # save resized image
    img = Image.open(os.path.join(root, source_dir, subpath))
    img = transforms(img)
    img.save(os.path.join(root, target_dir, subpath), quality=97)
    img.close()

def make_resized_dataset(root, source_dir, size, n_workers):
    root = os.path.expanduser(root)
    target_dir = 'CheXpert_{}x{}'.format(size, size)

    assert (not os.path.exists(os.path.join(root, target_dir, 'train.csv')) and \
            not os.path.exists(os.path.join(root, target_dir, 'valid.csv'))), 'Data exists at target dir.'
    print('Resizing dataset at root {}:\n\tsource {}\n\ttarget {}\n\tNew size: {}x{}'.format(root, source_dir, target_dir, size, size))

    transforms = T.Compose([T.Resize(size, Image.BICUBIC), T.CenterCrop(size)])


    for split in ['train', 'valid']:
        print('Processing ', split, ' split...')
        csv_path = os.path.join(root, source_dir, split + '.csv')

        # load data and preprocess NAs
        df = pd.read_csv(csv_path, keep_default_na=True)

        if split == 'train':
            # 1. fill NAs (blanks for unmentioned) as 0 (negatives)
            # attr columns ['No Finding', ..., 'Support Devices']; note AP/PA remains with NAs for Lateral pictures
            df[ChexpertDataset.attr_names] = df[ChexpertDataset.attr_names].fillna(0)

            # 2. fill -1 as 1 (U-Ones method described in paper)
            df[ChexpertDataset.attr_names] = df[ChexpertDataset.attr_names].replace(-1,1)

        # make new folders, resize image and store
        f = partial(_process_entry, root=root, source_dir=source_dir, target_dir=target_dir, transforms=transforms)
        with Pool(n_workers) as p:
            p.map(f, df['Path'].tolist())

        # replace `CheXpert-v1.0-small` root with new root defined above
        df['Path'] = df['Path'].str.replace(source_dir, target_dir)

        # save new df
        df.to_csv(os.path.join(root, target_dir, split + '.csv'))


# resize entire dataset
if False:
    root = '/mnt/disks/chexpert-ssd'
    source_dir = 'CheXpert-v1.0-small'
    new_size = 64
    n_workers = 16

    make_resized_dataset(root, source_dir, new_size, n_workers)

# compute dataset mean and std
if False:
    ds = ChexpertSmall(root=args.data_dir, train=True, transform=T.Compose([T.CenterCrop(320), T.ToTensor()]))
    m, s = compute_mean_and_std(ds)
    print('Dataset mean: {}; dataset std {}'.format(m, s))
    # Dataset mean: 0.533048452958796; dataset std 0.03490651403764978

# output a few images from the validation set and display labels
if False:
    import torchvision.transforms as T
    from torchvision.utils import save_image
    ds = ChexpertSmall(root=args.data_dir, train=False,
            transform=T.Compose([T.ToTensor(), T.Normalize(mean=[0.5330], std=[0.0349])]))
    print('Valid dataset loaded. Length: ', len(ds))
    for i in range(10):
        img, attr, patient_id = ds[i]
        save_image(img, 'test_valid_dataset_image_{}.png'.format(i), normalize=True, scale_each=True)
        print('Patient id: {}; labels: {}'.format(patient_id, attr))

