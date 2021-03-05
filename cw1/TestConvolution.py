import numpy as np
from MyConvolution import convolve
from scipy.signal import convolve2d
# import cv2

iterTest = 10

for i in range(iterTest):
    row, col = np.random.randint(10, 100), np.random.randint(10, 100)
    # testImage = np.random.rand(row, col, 3)
    testImage = np.random.rand(row, col)

    trow, tcol = np.random.choice([3, 5, 7]), np.random.choice([3, 5, 7])
    testKernel = np.random.randint(-5, 5, size=(trow, tcol))

    myconv = convolve(testImage, testKernel)

    # myconv = myconv[~np.all(myconv == 0, axis=(1, 2))]
    # myconv = myconv[:, ~np.all(myconv == 0, axis=(0, 2))]
    scipyconv = np.zeros(testImage.shape)
    scipyconv = convolve2d(testImage, testKernel, mode='same')
    # for j in range(3):
        # scipyconv[:, :, j] = convolve2d(testImage[:, :, j], testKernel, mode='same', boundary='fill', fillvalue=0)

    myconv = myconv.astype(np.float32)
    scipyconv = scipyconv.astype(np.float32)

    if (myconv == scipyconv).all():
        print(True)
    else:
        print(False)


