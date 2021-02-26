import numpy as np
from MyConvolution import convolve
from scipy.signal import convolve2d
# import cv2

iterTest = 20

for i in range(iterTest):
    row, col = np.random.randint(10, 500), np.random.randint(10, 500)
    testImage = np.random.rand(row, col, 3)

    trow, tcol = np.random.choice([3, 5, 7, 9, 11, 13, 15]), np.random.choice([3, 5, 7, 9, 11, 13, 15])
    testKernel = np.random.randint(-5, 5, size=(trow, tcol))

    myconv = convolve(testImage, testKernel)

    myconv = myconv[~np.all(myconv == 0, axis=(1, 2))]
    myconv = myconv[:, ~np.all(myconv == 0, axis=(0, 2))]
    scipyconv = []
    # scipyconv = convolve2d(testImage, testKernel, mode='valid')
    for j in range(3):
        scipyconv.append(convolve2d(testImage[:, :, j], testKernel, mode='valid'))

    scipyconv = np.array(scipyconv)
    # print(myconv)
    # print(scipyconv)

    if myconv.all() == scipyconv.all():
        print(True)
    else:
        print(False)

    

