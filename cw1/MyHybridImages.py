import math
import numpy as np

from MyConvolution import convolve

def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma: float) -> np.ndarray:
    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.
        :param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or colour
    shape=(rows,cols,channels))
        :type numpy.ndarray
    :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering lowImage :type float
    :param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
        :type numpy.ndarray
        :param highSigma: the standard deviation of the Gaussian used for low-pass filtering highImage
    before subtraction to create the high-pass filtered image
    :type float
    :returns returns the hybrid image created
    by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it with
    a high-pass image created by subtracting highImage from highImage convolved with
    a Gaussian of s.d. highSigma. The resultant image has the same size as the input images.
        :rtype numpy.ndarray
    """
    # All images are normalised
    lowfImage = convolve(lowImage/255, makeGaussianKernel(lowSigma))
    highfImage = (highImage / 255) - convolve(highImage/255, makeGaussianKernel(highSigma))

    return (lowfImage + highfImage) * 255 # rescale image to range 0-255



def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Use this function to create a 2D gaussian kernel with standard deviation sigma. The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
    """
    size = int(8.0 * sigma + 1.0)
    if (size % 2 == 0): size += 1

    # create 2D grid
    ax = np.linspace(-(size-1)/2, (size-1)/2, size)
    xx, yy = np.meshgrid(ax, ax)

    # 2D Gaussian equation
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)

    return kernel / np.sum(kernel) # normalise the kernel such that it sums to 1

