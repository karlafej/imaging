import cv2
import numpy as np


class Callback:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class PredictionsSaverCallback(Callback):
    def __init__(self, outpath, threshold):
        self.threshold = threshold
        self.outpath = outpath

    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] != "predict":
            return

        probs = kwargs['probs']
        files_name = kwargs['files_name']
        # Save the predictions
        for (pred, name) in zip(probs, files_name):
            name = name.split(".")[0]
            img_name = str(self.outpath/(name + ".bmp"))
            mask = pred > self.threshold
            mask = np.array(mask, dtype=np.uint8)
            cv2.imwrite(img_name, mask)

