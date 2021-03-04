import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
from pathlib import Path

from MyHybridImages import myHybridImages

file_dir = Path(__file__).resolve().parent

im1 = cv2.imread(str(file_dir/'data/dog.bmp'))
im2 = cv2.imread(str(file_dir/'data/cat.bmp'))

test = myHybridImages(im1, 16, im2, 8)
print(np.min(test), np.max(test), np.mean(test))

cv2.imwrite(str(file_dir/'hybrid.bmp'), test)
