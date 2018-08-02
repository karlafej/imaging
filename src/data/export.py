import cv2
import os
import re
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from functools import partial
import helpers


def folders_in(path_to_parent):
    '''
    list subfolders
    '''
    dir_path = Path(path_to_parent)
    if dir_path.is_dir():
        folders = list(str(x) for x in dir_path.iterdir() if x.is_dir())
    else:
        folders = []
    return folders

def get_DXA_lst(path_to_parent):
    '''
    find all folders to process

    returns folders' names and location flag
    '''
    DXA_lst = []
    if "DXA" in path_to_parent: #DXA
        DXA_lst.append(path_to_parent)
        where = -0
    else:
        folders = folders_in(path_to_parent)
        for folder in folders:
            if "DXA" in folder:   #Mouse
                DXA_lst.append(folder)
                where = -1
    if DXA_lst == []: # Mice !takes several minutes on primus
        for folder in folders:
            subfolders = folders_in(folder)
            for subfolder in subfolders:
                if "DXA" in subfolder:
                    DXA_lst.append(subfolder)
                    where = -2
    if DXA_lst == []:
        DXA_lst.append(path_to_parent)
        where = 0

    return DXA_lst, where

def mouse_part(number, start=1000, end=1900):
    '''
    split image list

    input: image number, limits
    output: name of the mouse part to use with the appropriate model
    '''
    if number < start:
        return 'start'
    elif number > end:
        return 'end'
    else: return 'middle'

def mouse_part_st(number, start=501, end=1700):
    '''
    split image list

    input: image number, limits
    output: name of the mouse part to use with the appropriate model
    '''
    if number < start:
        return 'st_start'
    elif number > end:
        return 'st_end'
    else: return 'st_middle'

def create_csv(inpath, datapath, mods=None, rec=False):
    '''
    create csv with image names, paths, numbers, split modifiers
    and save it in the datapath folder

    output: path to the CSV file, and list of folders to process
    '''
    if rec:
        DXA_lst, where = [inpath], 0
    else:
        DXA_lst, where = get_DXA_lst(inpath)
    df = pd.DataFrame(columns=("path", "img"))
    pattern = re.compile('.*?([0-9]+)$')
    scriptdir = str(Path(__file__).resolve().parent.parent.parent)
    cmd = ('Rscript '
           +scriptdir+'/startend/startPosition.R'+
           ' --mode="classify" --model="../startend/model.csv"'
           ' --p=0.3 --window=50 --paralell')
    for dxa in DXA_lst:
        tmpdf = pd.DataFrame()
        tmpdf['img'] = [img for img in os.listdir(dxa) if img.endswith('bmp')]
        tmpdf['path'] = dxa
        tmpdf['number'] = [int(m.group(1)) if m else None for m in (pattern.search(file[:-4]) for file in tmpdf['img'])]
        # call an external script to find the first image:
        inp = "".join([' --input="', dxa, '"'])
        cmd_line = cmd + inp
        proc = subprocess.run(cmd_line, stdout=subprocess.PIPE, shell=True)
        filename = (str(proc.stdout, 'utf-8')).strip()
        try:
            start = int(filename[-8:-4])
            n_st = 400
        except ValueError:
            start = tmpdf['number'].min()
            n_st = (1000 - start) if start < 1000 else 0
        print(dxa.split('/')[-1], "- start at image number: ", start)
        print("First image:", filename)
        tmpdf = pd.DataFrame()
        tmpdf['img'] = [img for img in os.listdir(dxa) if img.endswith('bmp')]
        tmpdf['path'] = dxa
        tmpdf['number'] = [int(m.group(1)) if m else None for m in (pattern.search(file[:-4]) for file in tmpdf['img'])]
        tmpdf = tmpdf.loc[tmpdf['number'] >= start]
        tmpdf.dropna(inplace=True)
        if mods is not None:
            if mods[0].startswith("st_"):
                tmpdf['split'] = tmpdf['number'].apply(mouse_part_st)
                #tmpdf.loc[((tmpdf['img'].str.contains('_M_')) & (tmpdf['split'] == "end")), 'split'] = "male_end"
                #tmpdf.loc[((tmpdf['img'].str.contains('_F_')) & (tmpdf['split'] == "end")), 'split'] = "female_end"
            else:
                tmpdf['split'] = tmpdf['number'].apply(mouse_part,
                                                       start=start+n_st,
                                                       end=start+n_st+900)
                tmpdf.loc[((tmpdf['img'].str.contains('_M_')) & (tmpdf['split'] == "end")), 'split'] = "male_end"
                tmpdf.loc[((tmpdf['img'].str.contains('_F_')) & (tmpdf['split'] == "end")), 'split'] = "female_end"
        df = pd.concat([df, tmpdf])

    df['ds'] = "test"
    CSV = datapath + 'predict_data.csv'
    df.to_csv(CSV)
    return CSV, DXA_lst, where

def create_dirs(dxa, predspath, where):
    '''
    create directory structure to put the output files in
    '''
    dirs = dxa.split('/')
    md = '/'.join(dirs[where:])
    pred = Path(predspath)
    (pred/md/"masks").mkdir(parents=True, exist_ok=True)
    (pred/md/"predictions").mkdir(parents=True, exist_ok=True)
    maskpath = str((pred/md/"masks").absolute()) + "/"
    outpath = str((pred/md/"predictions").absolute()) + "/"
    return maskpath, outpath

def process_mask(mask, imgpath, outpath, kernel):
    '''
    use the predicted mask on the original image and
    save the resulting file
    '''
    maskfile = str(mask)
    prefix = maskfile.split("/")[-1][:-4]
    imgloc = "".join([imgpath, "/", prefix, ".bmp"])
    imprefix = prefix[:-4] + '_' + prefix[-4:]

    with open(imgloc, 'rb') as fi:
        fi.seek(38)
        header = fi.read(8)

    img = cv2.imread(imgloc, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(maskfile, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    #mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    pred_masked = cv2.bitwise_and(img, img, mask=mask)
    outfile = "".join([outpath, imprefix, ".bmp"])
    cv2.imwrite(outfile, pred_masked)

    with open(outfile, 'r+b') as fo:
        fo.seek(38)
        fo.write(header)

@helpers.st_time(show_func_name=False)
def export_images(imgpath, maskpath, outpath, num_workers):
    '''
    process all images
    '''
    kernel = np.ones((2, 2), np.uint8)
    masks = list(Path(maskpath).glob("*png"))
    proc = partial(process_mask, imgpath=imgpath, outpath=outpath, kernel=kernel)

    pool = Pool(processes=num_workers)
    for _ in tqdm(pool.imap_unordered(proc, masks), total=len(masks), desc="Exporting"):
        pass
    pool.close()
    pool.join()
