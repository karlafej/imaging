import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from params import datapath
import pandas as pd

class DatasetFetcher:
    def __init__(self):
        """
            A tool used to automatically download, check, split and get
            relevant information on the dataset
        """
        self.train_data = None
        self.test_data = None
        self.train_masks_data = None
        self.train_files = None
        self.test_files = None
        self.train_masks_files = None
        self.csv = None

    def get_dataset(self,datapath=datapath):
        """
        Downloads the dataset and return the input paths
        Args:
            datapath : path to datasets
        Datasets should be located in directories datapath/train, datapath/train_masks,
        datapath/val, datapath/val_masks
        datapath/test

        Returns:
            list: [train_data, test_data, train_masks_data]

        """
       
        script_dir = os.path.dirname(os.path.abspath(__file__))
        destination_path = os.path.abspath(datapath)
        prefix = ""
       
        datasets_path = [destination_path + "/train" + prefix, 
                         destination_path + "/train_masks",
                         destination_path + "/val" + prefix,
                         destination_path + "/val_masks" + prefix,
                         destination_path + "/test_all" + prefix,
                         destination_path + "/data.csv"
                         ]
        is_datasets_present = True

        # Check if all folders are present
        for dir_path in datasets_path:
            if not os.path.exists(dir_path):
                is_datasets_present = False

        if not is_datasets_present:
                       print("Missing dataset")
        else:
            print("All datasets are present.")

        self.train_data = datasets_path[0]
        self.test_data = datasets_path[4]
        self.train_masks_data = datasets_path[1]
        self.val_data = datasets_path[2]
        self.val_masks_data = datasets_path[3]
        self.train_files = sorted(os.listdir(self.train_data))
        self.test_files = sorted(os.listdir(self.test_data))
        self.train_masks_files = sorted(os.listdir(self.train_masks_data))
        self.val_files = sorted(os.listdir(self.val_data))
        self.val_masks_files = sorted(os.listdir(self.val_masks_data))
        self.csv = pd.read_csv(datasets_path[5])
        return datasets_path

    def get_image_files(self, image_id, test_file=False, val_file=False, get_mask=False):
        if (get_mask & (not val_file)):
            if image_id + "_mask.png" in self.train_masks_files:
                return self.train_masks_data + "/" + image_id + "_mask.png"
            elif image_id + ".png" in self.train_masks_files:
                return self.train_masks_data + "/" + image_id + ".png"
            else:
                print(image_id)
                raise Exception("No mask with this ID found")
        elif (get_mask & val_file):
            if image_id + "_mask.png" in self.val_masks_files:
                return self.val_masks_data + "/" + image_id + "_mask.png"
            elif image_id + ".png" in self.val_masks_files:
                return self.val_masks_data + "/" + image_id + ".png"
            else:
                print(image_id)
                raise Exception("No mask with this ID found")
        elif test_file:
            if image_id + ".png" in self.test_files:
                return self.test_data + "/" + image_id + ".png"
        elif val_file:
            if image_id + ".png" in self.val_files:
                return self.val_data + "/" + image_id + ".png"
        else:
            if image_id + ".png" in self.train_files:
                return self.train_data + "/" + image_id + ".png"
        raise Exception("No image with this ID found")

    def get_image_matrix(self, image_path):
        img = Image.open(image_path)
        return np.asarray(img, dtype=np.uint8)

    def get_image_size(self, image):
        img = Image.open(image)
        return img.size

    def get_train_files(self, sample_size=None, part=None):
        """

        Args:
            sample_size (float, None):
                Value between 0 and 1 or None.
                Whether you want to have a sample of your dataset.

        Returns:
            list :
                Returns the dataset in the form:
                [train_data, train_masks_data, valid_data, valid_masks_data]
        """

        if part:
            train_ids = list(self.csv[(self.csv["ds"] == "train") & (self.csv["split"] ==                                                part)]["img"].str.split(".").str[0])
            val_ids = list(self.csv[(self.csv["ds"] == "val") & (self.csv["split"] ==                                                part)]["img"].str.split(".").str[0])
        # the val column is redundant... could be found from self.val_files
        # list(set(self.val_files).intersection(list(self.csv[...]))
        else:
            train_ids = list(map(lambda img: img.split(".")[0], self.train_files))
            val_ids = list(map(lambda img: img.split(".")[0], self.val_files))
            #train_ids = list(self.csv[(self.csv["ds"] == "train")]["img"].str.split(".").str[0])
            #val_ids = list(self.csv[(self.csv["ds"] == "val")]["img"].str.split(".").str[0])
                                    

        if sample_size:
            rnd = np.random.choice(train_ids, int(len(train_ids) * sample_size))
            train_ids = rnd.ravel()
            rnd =  np.random.choice(val_ids, int(len(val_ids) * sample_size))
            val_ids = rnd.ravel()

        train_ret = []
        train_masks_ret = []
        valid_ret = []
        valid_masks_ret = []

        for id in train_ids:
            if len(id)>3:
                train_ret.append(self.get_image_files(id))
                train_masks_ret.append(self.get_image_files(id, get_mask=True))

        for id in val_ids:
            if len(id)>3:
                valid_ret.append(self.get_image_files(id, val_file=True))
                valid_masks_ret.append(self.get_image_files(id, val_file=True, get_mask=True))

        return [np.array(train_ret).ravel(), np.array(train_masks_ret).ravel(),
                np.array(valid_ret).ravel(), np.array(valid_masks_ret).ravel()]

    def get_test_files(self, sample_size, part=None):
        if part:
            test_files = list(self.csv[(self.csv["ds"] == "test") & 
                                       (self.csv["split"] == part)]["img"])
        else:    
            test_files = self.test_files

        if sample_size:
            rnd = np.random.choice(test_files, int(len(test_files) * sample_size))
            test_files = rnd.ravel()

        ret = [None] * len(test_files)
        for i, file in enumerate(test_files):
            ret[i] = self.test_data + "/" + file

        return np.array(ret)
