import numpy as np
import torch.utils.data as data
from PIL import Image

import img.transformer as transformer


# Reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py#L66
class TrainImageDataset(data.Dataset):
    def __init__(self, X_data, y_data=None, img_resize=(128, 128),
                 X_transform=None, y_transform=None, threshold=0.5):
        """
            A dataset loader taking images paths as argument and return
            as them as tensors from getitem()

            Args:
                threshold (float): The threshold used to consider the mask present or not
                X_data (list): List of paths to the training images
                y_data (list, optional): List of paths to the target images
                img_resize (tuple): Tuple containing the new size of the images
                X_transform (callable, optional): A function/transform that takes in 2 numpy arrays.
                    Assumes X_data and y_data are not None.
                    (train_img, mask_img) and returns a transformed version with the same signature
                y_transform (callable, optional): A function/transform that takes in 2 numpy arrays.
                    Assumes X_data and y_data are not None.
                    (train_img, mask_img) and returns a transformed version with the same signature
        """
        self.threshold = threshold
        self.X_train = X_data
        self.y_train_masks = y_data
        self.img_resize = img_resize
        self.y_transform = y_transform
        self.X_transform = X_transform

    def __getitem__(self, index):
        """
            Args:
                index (int): Index
            Returns:
                tuple: (image, target) where target is class_index of the target class.
        """
        with open(self.X_train[index], 'rb') as f:
            img = Image.open(f)
        #img = Image.open(self.X_train[index])
            img = img.resize(self.img_resize, Image.ANTIALIAS)
            img = transformer.center_cropping_resize(img, self.img_resize)
            img = np.asarray(img.convert("RGB"), dtype=np.float32)

        # Pillow reads gifs
        with open(self.y_train_masks[index], 'rb') as f:
            mask = Image.open(f)
        #mask = Image.open(self.y_train_masks[index])
            mask = mask.resize(self.img_resize, Image.ANTIALIAS)
            mask = transformer.center_cropping_resize(mask, self.img_resize)
            mask = np.asarray(mask.convert("L"), dtype=np.float32)  # GreyScale

        if self.X_transform:
            img, mask = self.X_transform(img, mask)

        if self.y_transform:
            img, mask = self.y_transform(img, mask)

        img = transformer.image_to_tensor(img)
        mask = transformer.mask_to_tensor(mask, self.threshold)
        return img, mask

    def __len__(self):
        assert len(self.X_train) == len(self.y_train_masks)
        return len(self.X_train)


class TestImageDataset(data.Dataset):
    def __init__(self, X_data, img_resize=(128, 128)):
        """
            A dataset loader taking images paths as argument and return
            as them as tensors from getitem()
            Args:
                X_data (list): List of paths to the training images
                img_resize (tuple): Tuple containing the new size of the images
        """
        self.img_resize = img_resize
        self.X_train = X_data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_path = self.X_train[index]
        with open(img_path, 'rb') as f:  # strange pillow behavior with network drives...
            img = Image.open(f)
        #img = Image.open(img_path)
            img = img.resize(self.img_resize, Image.ANTIALIAS)
            img = transformer.center_cropping_resize(img, self.img_resize)
            img = np.asarray(img.convert("RGB"), dtype=np.float32)
            img = transformer.image_to_tensor(img)
        return img, img_path.split("/")[-1]

    def __len__(self):
        return len(self.X_train)
