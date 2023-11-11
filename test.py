import glog as log
import numpy as np
import math
from scipy.io import loadmat

from grayness_index import GraynessIndex


if __name__ == "__main__":
    gi = GraynessIndex()
    mat = loadmat("imgs/exampleimg.mat")
    gt = mat["gt"]
    input_im = mat["input_im"]
    pred_illum = gi.apply(input_im)
    error = np.arccos(pred_illum @ gt.transpose()) * 180 / math.pi
    log.info(f"Error: {error}")