import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

"""
Function #1: gaussian(window_size, sigma)

This function essentially generates a list of numbers (of length equal to window_size) 
sampled from a gaussian distribution. The sum of all the elements is equal to 1 and 
the values are normalized. Sigma is the standard deviation of the gaussian distribution.

Note: This is used to generate the 11x11 gaussian window
"""


def gaussian(window_size,sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


"""
Function #2: create_window(window_size, channel)
While we generated a 1D tensor of gaussian values, the 1D tensor itself is of no use to us. Hence we gotta convert it 
to a 2D tensor (the 11x11 Tensor we talked about earlier). The steps taken in this function are as follows,

Generate the 1D tensor using the gaussian function.
Convert it to a 2D tensor by cross-multiplying with its transpose (this preserves the gaussian character).
Add two extra dimensions to convert it to 4D. (This is only when SSIM is used as a loss function in computer vision)
Reshape to adhere to PyTorch weight’s format.

"""

# .mm normal mumtiplication of matricies
# .t is a transpose of matrix
def create_window(window_size,channel):
    _1D_window = gaussian(window_size,1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel,1,window_size,window_size).contiguous())
    return window



"""
Function #3: ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False)
Before we move onto the essentials, let us explore what happens in the function before the ssim metrics are calculated,
We set the maximum value of the normalized pixels (implementation detail; needn’t worry)
We initialize the gaussian window by means of the create_window() function IF a window was not provided during the function call.
Once these steps are completed, we go about calculating the various values (the sigmas and the mus of the world) 
which are needed to arrive at the final SSIM score.

"""

def _ssim(img1,img2,window,window_size,channel,size_average=True):
    """
        We first calculate μ(x), μ(y), their squares, and μ(xy). channels here store the number of color channels of the input image.
        The groups parameter is used to apply a convolution filter to all the input channels.
    """
    mu1 = F.conv2d(img1,window,padding=window_size//2,groups=channel)
    mu2 = F.conv2d(img2,window,padding=window_size//2,groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    """
    We first calculate μ(x), μ(y), their squares, and μ(xy). channels here store the number of color channels of the input image. 
    The groups parameter is used to apply a convolution filter to all the input channels. More information regarding groups can be found here.
    """

    sigma1_sq = F.conv2d(img1*img1,window,padding=window_size//2,groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2,window,padding=window_size//2,groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2,window,padding=window_size//2,groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    """
    Finally, we calculate the SSIM score and return the mean according to the formula 
    """

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# ssim : Structural Similarity Index mean

class SSIM(torch.nn.Module):

    def __init__(self,window_size=11,size_average = True):
        super(SSIM,self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size,self.channel)


    def forward(self,img1,img2):
        (_,channel,_,_) = img1.size()
        #torch.Size([3, 1, 11, 11])
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size,channel)
            #tensor.is_cuda
            # Is True if the Tensor is stored on the GPU, False otherwise.

            if img1.is_cuda:
                window = window.cuda(img1.get_device())

            # make sure the gaussian window have same type as window

            window = window.type_as(img1)

            self.window = window
            self.channel = channel
        return _ssim(img1,img2,window,self.window_size,channel,self.size_average)


def _logssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    ssim_map = (ssim_map - torch.min(ssim_map))/(torch.max(ssim_map)-torch.min(ssim_map))
    ssim_map = -torch.log(ssim_map + 1e-8)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class LOGSSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(LOGSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _logssim(img1, img2, window, self.window_size, channel, self.size_average)



def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)








