# Grayness Index

Unofficial Python implementation of the paper named ["On Finding Gray Pixels"](https://openaccess.thecvf.com/content_CVPR_2019/papers/Qian_On_Finding_Gray_Pixels_CVPR_2019_paper.pdf). 

Disclaimer:

*You may copy, distribute and modify the software provided that modifications are described and licensed for free under LGPL. Derivatives works (including modifications or anything statically linked to the library) can only be redistributed under LGPL, but applications that use the library don't have to be.*

## Installation

```
pip install grayness-index-python
```

## Basic Usage

```
import numpy as np
import math
from scipy.io import loadmat

from grayness_index import GraynessIndex

gi = GraynessIndex()
mat = loadmat("imgs/exampleimg.mat")
gt = mat["gt"]
input_im = mat["input_im"]  # np.ndarray with shape H x W x 3.
pred_illum = gi.apply(input_im)
error = np.arccos(pred_illum @ gt.transpose()) * 180 / math.pi
log.info(f"Error: {error}")
```

## References

[Official MatLab Code](https://github.com/yanlinqian/Grayness-Index)

To cite the original paper:

```
@inproceedings{qian2019cvpr,
  title={On Finding Gray Pixels},
  author={Qian, Yanlin and K{\"a}m{\"a}r{\"a}inen, Joni-Kristian and Nikkanen, Jarno and Matas, Jiri},
  booktitle={IEEE International Conference of Computer Vision and Pattern Recognition},
  year={2019}
}
```
