from pathlib import Path
import cv2
import pandas as pd
import os
import re
import numpy as np
import helpers
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

class Dxa:
    def __init__(self, inpath, predspath, mods=None, dxa=False):
        self.inpath = inpath 
        self.predspath = predspath
        self.mods = mods
        self.dxa = dxa

    def create_dirs(self, where):
        dirs = self.inpath.split('/')
        md = '/'.join(dirs[where:])
        pred = Path(self.predspath)
        (pred/md/"masks").mkdir(parents=True, exist_ok=True)
        (pred/md/"predictions").mkdir(parents=True, exist_ok=True)
        self.maskpath = str((pred/md/"masks").absolute()) + "/"
        self.outpath = str((pred/md/"predictions").absolute()) + "/"
        return self.maskpath, self.outpath


def folders_in(path_to_parent):
    dir_path = Path(path_to_parent)
    if dir_path.is_dir():
        folders = list(str(x) for x in dir_path.iterdir() if x.is_dir())
    else:
        folders = []
    return folders

def get_DXA_lst(path_to_parent):
    DXA_lst = []
    if "DXA" in path_to_parent: #DXA
        DXA_lst.append(path_to_parent)
        where = -0
    else:
        folders = folders_in(path_to_parent)
        for folder in folders:
            if "Rec" in folder:   #Mouse
                DXA_lst.append(folder)  
                where = -1
    if (DXA_lst == []): # Mice !takes several minutes on primus
        for folder in folders:
            subfolders = folders_in(folder)
            for subfolder in subfolders:
                if "Rec" in subfolder:
                    DXA_lst.append(subfolder)
                    where = -2
    return DXA_lst, where

def mouse_part(number, start = 1000, end = 1900):
        if number < start: return 'start'
        elif number > end: return 'end'
        else: return 'middle'
        
def mouse_part_st(number, start = 501, end = 1700):
        if number < start: return 'st_start'
        elif number > end: return 'st_end'
        else: return 'st_middle'

def create_csv(inpath, datapath, mods=None, dxa=False):
    if dxa:
        DXA_lst, where = [inpath], 0
    else:
        DXA_lst, where = get_DXA_lst(inpath)
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
    if mods is not None:
        if mods[0].startswith("st_"):
            df['split'] = df['number'].apply(lambda num: mouse_part_st(num))
        #df.loc[((df['img'].str.contains('_M_')) & (df['split'] == "end")), 'split'] = "male_end"
        #df.loc[((df['img'].str.contains('_F_')) & (df['split'] == "end")), 'split'] = "female_end"
        else:    
            df['split'] = df['number'].apply(lambda num: mouse_part(num))
            df.loc[((df['img'].str.contains('_M_')) & (df['split'] == "end")), 'split'] = "male_end"
            df.loc[((df['img'].str.contains('_F_')) & (df['split'] == "end")), 'split'] = "female_end"
    CSV = datapath + 'predict_data.csv'
    df.to_csv(CSV)
    return CSV, DXA_lst, where
        
def create_dirs(dxa, predspath, where):
    dirs = dxa.split('/')
    md = '/'.join(dirs[where:])
    pred = Path(predspath)
    (pred/md/"masks").mkdir(parents=True, exist_ok=True)
    (pred/md/"predictions").mkdir(parents=True, exist_ok=True)
    maskpath = str((pred/md/"masks").absolute()) + "/"
    outpath = str((pred/md/"predictions").absolute()) + "/"
    return maskpath, outpath

def process_mask(mask, imgpath, maskpath, outpath, kernel):
    maskfile = str(mask)
    prefix = maskfile.split("/")[-1][:-4]
    imgloc = "".join([imgpath, "/" ,prefix, ".bmp"])
    imprefix = prefix[:-4] + '_' + prefix[-4:]      
    
    with open(imgloc, 'rb') as fi:
        fi.seek(38)
        header = fi.read(8)
            
    img = cv2.imread(imgloc,cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(maskfile,cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    #mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 2)
    pred_masked = cv2.bitwise_and(img,img,mask = mask)
    outfile = "".join([outpath, imprefix, ".bmp"])
    cv2.imwrite(outfile, pred_masked)
                
    with open(outfile, 'r+b') as fo:
        fo.seek(38)
        fo.write(header)
        

@helpers.st_time(show_func_name=False)
def export_images(imgpath, maskpath, outpath, num_workers):         
    kernel = np.ones((2,2), np.uint8)
    masks = list(Path(maskpath).glob("*png"))
    proc = partial(process_mask, imgpath=imgpath, maskpath=maskpath, outpath=outpath, kernel=kernel)
    
    pool = Pool(processes=num_workers)
    for _ in tqdm(pool.imap_unordered(proc, masks), total=len(masks), desc="Exporting"):
        pass
    pool.close()    
    pool.join()        
    