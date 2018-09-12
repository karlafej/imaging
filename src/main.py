from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # load big images

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from pathlib import Path

import helpers
import img.augmentation as aug
import img.transformer as transformer
import params as par
from data.fetcher import DatasetFetcher
from data.dataset import TrainImageDataset, TestImageDataset
import nn.classifier
import nn.unet as unet
from nn.train_callbacks import TensorboardVisualizerCallback, TensorboardLoggerCallback, ModelSaverCallback
from nn.test_callbacks import PredictionsSaverCallback

def main(part=None):
    # Clear log dir first
    helpers.clear_logs_folder()

    # Hyperparameters
    img_resize = (700, 700)
    batch_size = 8
    #epochs = 4
    threshold = 0.5 # default 0.5
    sample_size = 0.05 #None # Put None to work on full dataset

    prefix = ""

    try:
        modelfiles = par.modelfiles
    except NameError as e:
        print(e)
        modelfiles = None

    if part:
        prefix = part + "_"
        if modelfiles:
            modelfile = modelfiles.get(part)
    else:
        modelfile = None

    # -- Optional parameters
    threads = 10
    use_cuda = torch.cuda.is_available()

    # Download the datasets
    ds_fetcher = DatasetFetcher(part=part)
    ds_fetcher.get_dataset()

    # Get the path to the files for the neural net
    X_train, y_train, X_valid, y_valid = ds_fetcher.get_train_files(sample_size=sample_size)
    full_x_test = ds_fetcher.get_test_files(sample_size=None)

    # -- Computed parameters
    # Get the original images size (assuming they are all the same size)
    #origin_img_size = ds_fetcher.get_image_size(X_train[0])
    # The image kept its aspect ratio so we need to recalculate the img size for the nn
    ### TO DO: different image sizes ###
    img_resize_centercrop = transformer.get_center_crop_size(X_train[0], img_resize)
    # Training callbacks
    tb_viz_cb = TensorboardVisualizerCallback(par.tb_viz_path)
    tb_logs_cb = TensorboardLoggerCallback(par.tb_logs_path)
    model_saver_cb = ModelSaverCallback(par.modelpath + "model_" + prefix + helpers.get_model_timestamp(),
                                        verbose=True)

    # Testing callbacks
    pred_saver_cb = PredictionsSaverCallback(outpath=Path(par.predspath), threshold=threshold)

    # Define our neural net architecture
    net = unet.UNet1024((3, *img_resize_centercrop))
    classifier = nn.classifier.ImgClassifier(net, par.epochs)

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
        classifier.restore_model("".join([par.modelpath, modelfile]))
        print(f'Restoring model {modelfile}')

    classifier.train(train_loader, valid_loader, par.epochs,
                     callbacks=[tb_viz_cb, tb_logs_cb, model_saver_cb])

    test_ds = TestImageDataset(full_x_test, img_resize)
    test_loader = DataLoader(test_ds, batch_size,
                             sampler=SequentialSampler(test_ds),
                             num_workers=threads,
                             pin_memory=use_cuda)

    # Predict & save
    classifier.predict(test_loader, callbacks=[pred_saver_cb])


if __name__ == "__main__":
    if par.mods:
        for mpart in par.mods:
            print("Using {} part." .format(mpart))
            main(part=mpart)
    else:
        main()
