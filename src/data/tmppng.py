import pandas as pd
import re
from export import get_DXA_lst
import cv2
import os

def create_bmplist(inpath): #coz neni zrovna elegantni, ale predelat to muzu vzdycky
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
    df['full'] = df['path'].str.cat(df['img'], sep='/')
    files = list(df['full'])
    return files

def save_png(bmpfile):
    img = cv2.imread(bmpfile)
    cv2.imwrite("".join([bmpfile[:-3], "png"]), img)
