import numpy as np
from scipy.signal import convolve2d

def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray: 
    """
    Convolve an image with a kernel assuming zero-padding of the image to handle the borders
        :param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
        :type numpy.ndarray
        :param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
        :type numpy.ndarray
    :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray
    """
    # Your code here. You'll need to vectorise your implementation to ensure it runs # at a reasonable speed.
    kernel = np.flip(kernel)

    cimage = np.zeros(image.shape)
    rows, cols = image.shape
    trows, tcols = kernel.shape
    tr, tc = trows//2, tcols//2

    for x in range(tc, cols-tc):
        for y in range(tr, rows-tr):
            cimage[y, x] = (kernel * image[y-tr:y+tr+1, x-tc:x+tc+1]).sum()

    return cimage