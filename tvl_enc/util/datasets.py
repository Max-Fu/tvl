# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import tarfile
from PIL import Image
import numpy as np
import math

import torch
from torchvision import datasets, transforms
from torch.utils import data
from torch.utils.data import DistributedSampler

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class DistributedImportanceSampler(DistributedSampler):
    def __init__(self, dataset, sampling_ratios, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.sampling_ratios = sampling_ratios
        assert isinstance(dataset, data.ConcatDataset), "DistributedImportanceSampler only works with ConcatDataset"
        assert len(dataset.datasets) == len(sampling_ratios), "Number of datasets and sampling ratios must match"
        assert all([1>=r>0 for r in sampling_ratios]), "Sampling ratios must be between 0 and 1"
        self.dataset_lengths = [len(d) for d in self.dataset.datasets]
        self.individual_size = [int(l * r) for l, r in zip(self.dataset_lengths, sampling_ratios)]
        self.cumulative_sizes = self.dataset.cumulative_sizes

        total_size = sum(self.individual_size)
        if self.drop_last and total_size % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (total_size - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(total_size / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

        # print statistics
        original_sizes = [i for i in zip(self.dataset.datasets, self.dataset_lengths)]
        final_sizes = [i for i in zip(self.dataset.datasets, self.individual_size)]
        print("Dataset Statistics:")
        print("-----------------------------------------------------------------------")
        print("{:<20} {:<15} {:<15} {:<10}".format('Dataset', 'Original Size', 'Final Size', '% of Total'))
        print("-----------------------------------------------------------------------")
        for ((dataset, original), (_, final)) in zip(original_sizes, final_sizes):
            dataset_name = type(dataset).__name__  # Assuming each dataset has a different class
            percentage = final / self.total_size * 100
            print("{:<20} {:<15} {:<15} {:<10.2f}".format(dataset_name, original, final, percentage))
        print("-----------------------------------------------------------------------")


    def get_dataset_indices(self, subdataset : data.Dataset, base_idx=0):
        """
        helper function that returns the indices of a subdataset, given its base index
        """
        if self.shuffle: 
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(subdataset), generator=g).tolist()
        else:
            indices = list(range(len(subdataset)))
        for i in range(len(indices)):
            indices[i] += base_idx
        return indices

    def __iter__(self):
        indices = []
        for i, (dataset, data_size) in enumerate(zip(self.dataset.datasets, self.individual_size)):
            if i == 0:
                base_idx = 0
            else:
                base_idx = self.cumulative_sizes[i-1]
            indices += self.get_dataset_indices(dataset, base_idx)[:data_size]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # shuffle again as indices come in the order of datasets 
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(indices), generator=g).tolist()

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_type == 'folder':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
    elif args.data_type == 'tar':
        root = os.path.join(args.data_path, 'train.tar' if is_train else 'val.tar')
        dataset = ImageTarDataset(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)



class ImageTarDataset(data.Dataset):
  def __init__(self, tar_file, return_labels=True, transform=transforms.ToTensor()):
    '''
    return_labels:
    Whether to return labels with the samples
    transform:
    A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
    '''
    self.tar_file = tar_file
    self.tar_handle = None
    categories_set = set()
    self.tar_members = []
    self.categories = {}
    self.categories_to_examples = {}
    with tarfile.open(tar_file, 'r:') as tar:
      for index, tar_member in enumerate(tar.getmembers()):
        if tar_member.name.count('/') != 2:
          continue
        category = self._get_category_from_filename(tar_member.name)
        categories_set.add(category)
        self.tar_members.append(tar_member)
        cte = self.categories_to_examples.get(category, [])
        cte.append(index)
        self.categories_to_examples[category] = cte
    categories_set = sorted(categories_set)
    for index, category in enumerate(categories_set):
      self.categories[category] = index
    self.num_examples = len(self.tar_members)
    self.indices = np.arange(self.num_examples)
    self.num = self.__len__()
    print("Loaded the dataset from {}. It contains {} samples.".format(tar_file, self.num))
    self.return_labels = return_labels
    self.transform = transform

  def _get_category_from_filename(self, filename):
    begin = filename.find('/')
    begin += 1
    end = filename.find('/', begin)
    return filename[begin:end]

  def __len__(self):
    return self.num_examples

  def __getitem__(self, index):
    index = self.indices[index]
    if self.tar_handle is None:
      self.tar_handle = tarfile.open(self.tar_file, 'r:')

    sample = self.tar_handle.extractfile(self.tar_members[index])
    image = Image.open(sample).convert('RGB')
    image = self.transform(image)

    if self.return_labels:
      category = self.categories[self._get_category_from_filename(
          self.tar_members[index].name)]
      return image, category
    return image