import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data.dataset import TrainImageDataset
from data.fetcher import DatasetFetcher
from params import *

import numpy as np
import os
import pdb



def main(part=None):
    
    img_resize = (700, 700)
    batch_size = 64
    sample_size = None
    threads = 20
    threshold = 0.5
    
    ds_fetcher = DatasetFetcher()
    ds_fetcher.get_dataset()

    # Get the path to the files for the neural net
    X_train, y_train, X_valid, y_valid = ds_fetcher.get_train_files(sample_size=sample_size, part=part)
    full_x_test = ds_fetcher.get_test_files(sample_size=None, part=part)
        
    train_ds = TrainImageDataset(X_train, y_train, img_resize)
    train_loader = DataLoader(train_ds, batch_size,
                              shuffle=True,
                              num_workers=threads)

    valid_ds = TrainImageDataset(X_valid, y_valid, img_resize, threshold=threshold)
    valid_loader = DataLoader(valid_ds, batch_size,
                              shuffle=True,
                              num_workers=threads)
    
    
    print("Train dataset: {}, validation dataset: {} samples "
          .format(len(train_loader.dataset), len(valid_loader.dataset)))

    ipop_mean = []
    ipop_std0 = []
    ipop_std1 = []
    i_mask_pop_mean = []
    i_mask_pop_std0 = []
    i_mask_pop_std1 = []
    for i, (images, masks) in enumerate(valid_loader):
        numpy_image = images.numpy()  
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
        ipop_mean.append(batch_mean)
        ipop_std0.append(batch_std0)
        ipop_std1.append(batch_std1)
        
        numpy_image = masks.numpy()
        batch_mean = np.mean(numpy_image, axis=(0,1,2))
        batch_std0 = np.std(numpy_image, axis=(0,1,2))
        batch_std1 = np.std(numpy_image, axis=(0,1,2), ddof=1)
        i_mask_pop_mean.append(batch_mean)
        i_mask_pop_std0.append(batch_std0)
        i_mask_pop_std1.append(batch_std1)

    ipop_mean = np.array(ipop_mean).mean(axis=0)
    ipop_std0 = np.array(ipop_std0).mean(axis=0)
    ipop_std1 = np.array(ipop_std1).mean(axis=0)
    i_mask_pop_mean = np.array(i_mask_pop_mean).mean(axis=0)
    i_mask_pop_std0 = np.array(i_mask_pop_std0).mean(axis=0)
    i_mask_pop_std1 = np.array(i_mask_pop_std1).mean(axis=0)

    print(4*'#', 'TRAIN IMG VALUES', 4*'#')
    print("Mean:", ipop_mean.tolist())
    print("Standard deviation:", ipop_std0.tolist())
    print("Standard deviation (ddof=1):", ipop_std1.tolist())

    print(4*'#', 'TRAIN MASK VALUES', 4*'#')
    print("Mean:", i_mask_pop_mean.tolist())
    print("Standard deviation:", i_mask_pop_std0.tolist())
    print("Standard deviation (ddof=1):", i_mask_pop_std1.tolist())

    vpop_mean = []
    vpop_std0 = []
    vpop_std1 = []
    v_mask_pop_mean = []
    v_mask_pop_std0 = []
    v_mask_pop_std1 = []
    for i, (images, masks) in enumerate(valid_loader):
        numpy_image = images.numpy()  
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
        vpop_mean.append(batch_mean)
        vpop_std0.append(batch_std0)
        vpop_std1.append(batch_std1)
        
        numpy_image = masks.numpy()
        batch_mean = np.mean(numpy_image, axis=(0,1,2))
        batch_std0 = np.std(numpy_image, axis=(0,1,2))
        batch_std1 = np.std(numpy_image, axis=(0,1,2), ddof=1)
        v_mask_pop_mean.append(batch_mean)
        v_mask_pop_std0.append(batch_std0)
        v_mask_pop_std1.append(batch_std1)

    vpop_mean = np.array(vpop_mean).mean(axis=0)
    vpop_std0 = np.array(vpop_std0).mean(axis=0)
    vpop_std1 = np.array(vpop_std1).mean(axis=0)
    v_mask_pop_mean = np.array(v_mask_pop_mean).mean(axis=0)
    v_mask_pop_std0 = np.array(v_mask_pop_std0).mean(axis=0)
    v_mask_pop_std1 = np.array(v_mask_pop_std1).mean(axis=0)

    print(4*'#', 'VALID IMG VALUES', 4*'#')
    print("Mean:", vpop_mean.tolist())
    print("Standard deviation:", vpop_std0.tolist())
    print("Standard deviation (ddof=1):", vpop_std1.tolist())

    print(4*'#', 'VALID MASK VALUES', 4*'#')
    print("Mean:", v_mask_pop_mean.tolist())
    print("Standard deviation:", v_mask_pop_std0.tolist())
    print("Standard deviation (ddof=1):", v_mask_pop_std1.tolist())
    
if __name__ == "__main__":
    if mods:
        for part in mods:
            print("Using {} part." .format(part))
            main(part=part)
    else:
        main()