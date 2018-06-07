from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # load big images
import nn.classifier
import nn.unet as unet

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from data.fetcher import DatasetFetcher
import nn.classifier
from nn.test_callbacks import PredictionsSaverCallback
from params import *


from data.dataset import TestImageDataset
import img.transformer as transformer

from data.export import *
import sys, getopt
from tqdm import tqdm

def main(argv):
  
    # Hyperparameters
    img_resize = (700, 700)
    batch_size = 12
    epochs = 15
    threshold = 0.5 # default 0.5
    sample_size = None # Put None to work on full dataset
     

    # -- Optional parameters
    threads = 8
    use_cuda = torch.cuda.is_available() 

    # -- inpath from command line
    
    try:
        opts, args = getopt.getopt(argv,"hi:",["input="])
    except getopt.GetoptError:
        print ('argv[0] -i <inputpath>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('argv[0] -i <inputpath>')
            sys.exit()
        elif opt in ("-i", "--input"):
            inpath = arg
            
    if mods:
        CSV, dxa_lst, where = create_csv(inpath, datapath, mods)

    predspath = inpath #TESTTESTTEST 
    
    for dxa in dxa_lst:
        print("..." + dxa[-30:]) 
        maskpath, outpath = create_dirs(dxa, predspath, where)
        
        if mods:
            for part in mods:
                
                modelfile = modelfiles[part]
                ds_fetcher = DatasetFetcher()
                ds_fetcher.get_dataset(datapath=dxa, csv = CSV, prod = True)
                full_x_test = ds_fetcher.get_test_files(sample_size=None, part=part, prod = True)
                if len(full_x_test) > 0:
                    print(part)
                    img_resize_centercrop = transformer.get_center_crop_size(full_x_test[0], img_resize)
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
           
        export_images(imgpath=dxa, maskpath=maskpath, outpath=outpath, num_workers=threads)

if __name__ == "__main__":
    main(sys.argv[1:])
