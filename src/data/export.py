from pathlib import Path
import cv2
import pandas as pd
import os
import re
import numpy as np
import helpers
from tqdm import tqdm

def folders_in(path_to_parent):
    dir_path = Path(path_to_parent)
    if dir_path.is_dir():
        folders = list(str(x) for x in dir_path.iterdir() if x.is_dir())
    else:
        folders = []
    return folders

def get_DXA_lst(path_to_parent):
    DXA_lst = []
    if "RecDXA" in path_to_parent: #DXA
        DXA_lst.append(path_to_parent)
    else:
        folders = folders_in(path_to_parent)
        for folder in folders:
            if "RecDXA" in folder:   #Mouse
                DXA_lst.append(folder)
    
    if (DXA_lst == []): # Mice !takes several minutes on primus
        for folder in folders:
            subfolders = folders_in(folder)
            for subfolder in subfolders:
                if "RecDXA" in subfolder:
                    DXA_lst.append(subfolder)
    return DXA_lst

def mouse_part(number, start = 1000, end = 1900):
        if number < start: return 'start'
        elif number > end: return 'end'
        else: return 'middle'

def create_csv(inpath, datapath):
    DXA_lst = get_DXA_lst(inpath)
    df = pd.DataFrame(columns=("path", "img"))
    for dxa in DXA_lst:
        tmpdf = pd.DataFrame()
        tmpdf['img'] = [img for img in os.listdir(dxa) if img.endswith('bmp')]
        tmpdf['path'] = dxa
        df = pd.concat([df, tmpdf])
    pattern = re.compile('.*?([0-9]+)$')
    df['number'] = [int(m.group(1)) if m else None for m in (pattern.search(file[:-4]) for file in df['img'])]
    df.dropna(inplace=True)
    df['ds'] = "test"
    df['split'] = df['number'].apply(lambda num: mouse_part(num))
    CSV = datapath + 'data.csv'
    df.to_csv(CSV)
    return CSV, DXA_lst
        
def create_dirs(dxa, predspath):
    dirs = dxa.split('/')
    md = '/'.join(dirs[-3:])
    pred = Path(predspath)
    (pred/md/"masks").mkdir(parents=True, exist_ok=True)
    (pred/md/"predictions").mkdir(parents=True, exist_ok=True)
    maskpath = str((pred/md/"masks").absolute()) + "/"
    outpath = str((pred/md/"predictions").absolute()) + "/"
    return maskpath, outpath

@helpers.st_time(show_func_name=False)
def export_images(imgpath, maskpath, outpath):         
    kernel = np.ones((2,2), np.uint8)
    masks = list(Path(maskpath).glob("*png"))
    it_count = len(masks)
    
    with tqdm(total=it_count, desc="Exporting") as pbar:
        for m in masks:
            maskfile = str(m)
            prefix = maskfile.split("/")[-1][:-4]
            imgloc = "".join([imgpath, "/" ,prefix, ".bmp"])
            
            with open(imgloc, 'rb') as fi:
                fi.seek(38)
                header = fi.read(8)
            
            img = cv2.imread(imgloc,cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(maskfile,cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask = cv2.dilate(mask, kernel, iterations = 2)
            pred_masked = cv2.bitwise_and(img,img,mask = mask)
            outfile = "".join([outpath, prefix, ".bmp"])
            cv2.imwrite(outfile, pred_masked)
            
            with open(outfile, 'r+b') as fo:
                fo.seek(38)
                fo.write(header)  
                
            pbar.update(1)