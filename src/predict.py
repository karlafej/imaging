import sys
import getopt

from PIL import ImageFile
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from pathlib import Path
from datetime import date

import nn.classifier
import nn.unet as unet
from nn.test_callbacks import PredictionsSaverCallback
from data.fetcher import DatasetFetcher
from data.dataset import TestImageDataset
from data.export import create_csv, create_dirs, export_images, get_DXA_lst
import img.transformer as transformer
from params import modelfiles, modelpath, datapath
from helpers import print_help

ImageFile.LOAD_TRUNCATED_IMAGES = True # load big images


def main(argv):

    # Hyperparameters
    img_resize = (700, 700)
    batch_size = 12
    epochs = 15
    threshold = 0.5 # default 0.5
    #sample_size = None # Put None to work on full dataset

    # -- Optional parameters
    threads = 8
    use_cuda = torch.cuda.is_available()

    try:
        opts, args = getopt.getopt(argv, "hi:o:c:f:sdr", ["input=", "output=", "csv=", "rec"])
    except getopt.GetoptError:
        print ('argv[0] -i <inputpath> -s')
        sys.exit(2)
    stretched = False
    output = None
    csvfile = None

    for opt, arg in opts:
        if opt == '-h':
            print_help(argv[0])
            sys.exit()
        elif opt in ("-i", "--input"):
            inpath = arg
        elif opt in ("-c", "--csv"):
            csvfile = arg
        elif opt in ("-s", "--str"):
            stretched = True
        elif opt in ("-o", "--output"):
            output = arg


    if stretched:
        mods = ["st_start", "st_middle", "st_end"]
    else:
        mods = ["start", "middle", "male_end", "female_end"]

    dxa_lst, where = get_DXA_lst(inpath)

    if mods: # but mods are everytime!
        if csvfile is not None:
            CSV = csvfile
        else:
            CSV = create_csv(datapath, DXA_lst=dxa_lst, mods=mods)

    if output is not None:
        predspath = output
    else:
        predspath = inpath

    for dxa in dxa_lst:
        print("..." + dxa[-30:])
        maskpath, outpath = create_dirs(dxa, predspath, where)

        if mods:
            for part in mods:

                modelfile = modelfiles[part]
                ds_fetcher = DatasetFetcher(part=part)
                ds_fetcher.get_dataset(data_path=dxa, csv=CSV, predicting=True)
                full_x_test = ds_fetcher.get_test_files(sample_size=None, predicting=True)
                if full_x_test.size != 0:
                    print(part)
                    img_resize_centercrop = transformer.get_center_crop_size(full_x_test[0],
                                                                             img_resize)
                    net = unet.UNet1024((3, *img_resize_centercrop))

                    classifier = nn.classifier.ImgClassifier(net, epochs)
                    classifier.restore_model("".join([modelpath, modelfile]))

                    test_ds = TestImageDataset(full_x_test, img_resize)
                    test_loader = DataLoader(test_ds, batch_size,
                                             sampler=SequentialSampler(test_ds),
                                             num_workers=threads,
                                             pin_memory=use_cuda)

                    pred_saver_cb = PredictionsSaverCallback(outpath=maskpath, threshold=threshold)
                    classifier.predict(test_loader, callbacks=[pred_saver_cb])

        dxa = Path(dxa)
        export_images(imgpath=dxa, maskpath=maskpath, outpath=outpath, num_workers=threads)

        out = Path(outpath)
        fname = dxa.name + '.log'
        logfile = out/fname
        models = "\n".join([modelfiles[key] for key in mods])
        logfile.write_text(str(date.today()) + "\nModel:\n" + models)

    print("*** FINISHED ***")

if __name__ == "__main__":
    main(sys.argv[1:])
