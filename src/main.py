from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # load big images
import nn.classifier
import nn.unet as unet
import helpers

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import img.augmentation as aug
from data.fetcher import DatasetFetcher
import nn.classifier
from nn.train_callbacks import TensorboardVisualizerCallback, TensorboardLoggerCallback, ModelSaverCallback
from nn.test_callbacks import PredictionsSaverCallback
from params import *

import os

from data.dataset import TrainImageDataset, TestImageDataset
import img.transformer as transformer


def main(part=None):
    # Clear log dir first
    helpers.clear_logs_folder()

    # Hyperparameters
    img_resize = (700, 700)
    batch_size = 8
    epochs = 4
    threshold = 0.5 # default 0.5
    sample_size = None # Put None to work on full dataset
    if part:
        modelfile = modelfiles[part]
    
    prefix = ""
    if part:
        prefix = part + "_"
    
    #modelfile = "model_male_end_2018-04-19_17h04"
    
   
    # -- Optional parameters
    threads = 10
    use_cuda = torch.cuda.is_available()
  
    # Download the datasets
    ds_fetcher = DatasetFetcher()
    ds_fetcher.get_dataset()

    # Get the path to the files for the neural net
    X_train, y_train, X_valid, y_valid = ds_fetcher.get_train_files(sample_size=sample_size, part=part)
                                                                
    full_x_test = ds_fetcher.get_test_files(sample_size=None, part=part)

    # -- Computed parameters
    # Get the original images size (assuming they are all the same size)
    #origin_img_size = ds_fetcher.get_image_size(X_train[0])
    # The image kept its aspect ratio so we need to recalculate the img size for the nn
    ### TO DO: different image sizes ###
    img_resize_centercrop = transformer.get_center_crop_size(X_train[0], img_resize) 
    # Training callbacks
    tb_viz_cb = TensorboardVisualizerCallback(tb_viz_path)
    tb_logs_cb = TensorboardLoggerCallback(tb_logs_path)
    model_saver_cb = ModelSaverCallback(modelpath + "model_" + prefix + helpers.get_model_timestamp(), verbose=True)
  
    # Testing callbacks
    pred_saver_cb = PredictionsSaverCallback(outpath=predspath, threshold=threshold)

    # Define our neural net architecture
    net = unet.UNet1024((3, *img_resize_centercrop))
    classifier = nn.classifier.ImgClassifier(net, epochs)
    
    train_ds = TrainImageDataset(X_train, y_train, img_resize, X_transform=aug.augment_img)
    train_loader = DataLoader(train_ds, batch_size,
                              sampler=RandomSampler(train_ds),
                              num_workers=threads,
                              pin_memory=use_cuda)

    valid_ds = TrainImageDataset(X_valid, y_valid, img_resize, threshold=threshold)
    valid_loader = DataLoader(valid_ds, batch_size,
                              sampler=RandomSampler(valid_ds),
                              num_workers=threads,
                              pin_memory=use_cuda)
  
    print("Training on {} samples and validating on {} samples "
          .format(len(train_loader.dataset), len(valid_loader.dataset)))

    if modelfile:
        classifier.restore_model("".join([modelpath, modelfile]))
    
    classifier.train(train_loader, valid_loader, epochs,
                     callbacks=[tb_viz_cb, tb_logs_cb, model_saver_cb])

    test_ds = TestImageDataset(full_x_test, img_resize)
    test_loader = DataLoader(test_ds, batch_size,
                             sampler=SequentialSampler(test_ds),
                             num_workers=threads,
                             pin_memory=use_cuda)

    # Predict & save
    classifier.predict(test_loader, callbacks=[pred_saver_cb])

    
if __name__ == "__main__":
    if mods:
        for part in mods:
            print("Using {} part." .format(part))
            main(part=part)
    else:
        main()
