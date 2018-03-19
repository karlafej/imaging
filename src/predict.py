from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # load big images
import nn.classifier
import nn.unet as unet

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from data.fetcher import DatasetFetcher
import nn.classifier
from nn.test_callbacks import PredictionsSaverCallback
from params import *


from data.dataset import TestImageDataset
import img.transformer as transformer

def main(part=None):
  
    # Hyperparameters
    img_resize = (700, 700)
    batch_size = 12
    epochs = 15
    threshold = 0.5 # default 0.5
    sample_size = None # Put None to work on full dataset
    modelfile = "model_2018-03-13_12h37"
   
    # -- Optional parameters
    threads = 10
    use_cuda = torch.cuda.is_available()
  

    # Download the datasets
    ds_fetcher = DatasetFetcher()
    ds_fetcher.get_dataset()
    
    X_train, y_train, X_valid, y_valid = ds_fetcher.get_train_files(sample_size=0.01, part=part)
    full_x_test = ds_fetcher.get_test_files(sample_size=None, part=part)
    
    img_resize_centercrop = transformer.get_center_crop_size(X_train[0], img_resize)
  
    # Testing callbacks
    pred_saver_cb = PredictionsSaverCallback(outpath=predspath, threshold=threshold)

    # Define our neural net architecture
    net = unet.UNet1024((3, *img_resize_centercrop))
    classifier = nn.classifier.ImgClassifier(net, epochs)
  

    classifier.restore_model("".join([modelpath, modelfile]))
  
    test_ds = TestImageDataset(full_x_test, img_resize)
    test_loader = DataLoader(test_ds, batch_size,
                             sampler=SequentialSampler(test_ds),
                             num_workers=threads,
                             pin_memory=use_cuda)

    # Predict & save
    classifier.predict(test_loader, callbacks=[pred_saver_cb])


if __name__ == "__main__":
    main()
