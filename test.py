import math
import numpy as np
import torch
import glog as log

from scipy.io import loadmat

from grayness_index import GraynessIndex


if __name__ == "__main__":
    gi = GraynessIndex()
    mat = loadmat("imgs/exampleimg.mat")
    gt = mat["gt"]
    input_im = mat["input_im"]

    log.info("Numpy array test:")
    pred_illum = gi.apply(input_im)
    error = np.arccos(pred_illum @ gt.transpose()) * 180 / math.pi
    log.info(f"Error: {error}")

    log.info("Torch tensor test:")
    input_im_T = torch.tensor(input_im)
    input_im_T = torch.stack([input_im_T, input_im_T], dim=0)
    pred_illum = gi.apply(input_im_T)
    error = np.arccos(pred_illum.cpu().numpy()[0] @ gt.transpose()) * 180 / math.pi
    log.info(f"Error: {error}")
