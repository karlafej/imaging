import cv2


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
            img_name = "".join([self.outpath, name, ".png"])
            cv2.imwrite(img_name, pred)
           
        