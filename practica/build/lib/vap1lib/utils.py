import matplotlib.pyplot as plt
import skimage as ski
import numpy as np
import math
import scipy.stats as st
from scipy import ndimage
import cv2
from skimage import io
import cv2
from skimage import img_as_float, color

class Utils:

    @staticmethod
    def image_gray(url):
        img = io.imread(url, as_gray=True)
        if(img is not None):
            return np.array(img)

    @staticmethod
    def img_save(file_name, img):
        io.imsave(file_name, img)


    @staticmethod
    def get_kernel_shape(kernel):
        try: 
            (kerX, kerY) = kernel.shape
        except:
            try:
                kernel = np.array([kernel])
                (kerX, kerY) = kernel.shape
            except:
                raise Exception("Kernel shape error ",kernel.shape)
        return (kerX, kerY)

    @staticmethod
    def copy_black_frame(inImage, kCenter=(), kShape=()):

        if not kShape:
            raise Exception("Not shape provided")
        
        (kerX, kerY) = kShape
        (imgX, imgY) = inImage.shape

        if not kCenter:
            kCenter = (int(np.floor(kerX/2)), int(np.floor(kerY/2)))

        copiaB = np.zeros((kerX+imgX-1, kerY+imgY-1))
        print(copiaB.shape)
        copiaB[kCenter[0]:kCenter[0]+imgX, kCenter[1]:kCenter[1]+imgY] = inImage
        return  copiaB

    @staticmethod
    def plot_gray_color(gray_img, color_img):
        
        corners_float = img_as_float(color_img)
        corner_rgb = color.gray2rgb(corners_float)
        corners = corner_rgb * [1,0,0]

        img_float = img_as_float(gray_img)
        img_rgb = color.gray2rgb(img_float)
        img = img_rgb

        _, axarr = plt.subplots(ncols=1, figsize=(50,50), sharex=True, sharey=True)
        axarr.imshow(img+corners, vmin = 0.0, vmax= 1.0)
        plt.savefig("harris.png")
        plt.show()

    @staticmethod
    def plt_color(color_img, output):
        corners_float = img_as_float(color_img)
        corner_rgb = color.gray2rgb(corners_float)
        corners = corner_rgb * [1,0,0]

        _, axarr = plt.subplots(ncols=1, figsize=(50,50), sharex=True, sharey=True)
        axarr.imshow(corners, vmin = 0.0, vmax= 1.0)
        plt.savefig(output)
        plt.show()
    @staticmethod
    def plot_img(img):
        plt.imshow(img, cmap='gray', vmin = 0.0, vmax= 1.0)
        plt.show()

    @staticmethod
    def plot_test_compare(original_img, a_img, b_img, plt_hist=True):
        ## imShow asegurar que las imagenes se centran vmin vmax Cambiar -> [0.0, 1.0]
        """
        Parameters
        ----------
        original_img: array
        Imagen original.
        a_img: array
        Imagen resultado del primer algoritmo a comparar
        b_img: array
        Imagen resultado del segundo algoritmo a comparar
        """

        if plt_hist:
            kwargs = dict(histtype='stepfilled', alpha=0.3, bins=40)

            plt.hist(original_img.flat, **kwargs)
            plt.hist(a_img.flat, **kwargs)
            plt.hist(b_img.flat, **kwargs)
            plt.show()

        _, axarr = plt.subplots(ncols=3, figsize=(50,50), sharex=True, sharey=True)
        axarr[0].imshow(original_img, cmap='gray', vmin = 0.0, vmax= 1.0)
        axarr[1].imshow(a_img, cmap='gray', vmin = 0.0, vmax= 1.0)
        axarr[2].imshow(b_img, cmap='gray', vmin = 0.0, vmax= 1.0)
        plt.show()

    @staticmethod
    def plot_test_compare_n(list_imgs, cols):

        kwargs = dict(histtype='stepfilled', alpha=0.3, bins=40)

        for img in list_imgs:
            plt.hist(img[1].flat, **kwargs)

        plt.show()
        rows = math.floor(len(list_imgs)/cols)
        auxr = rows

        while auxr > 0:
            _, axarr = plt.subplots(ncols=cols,figsize=(40,40),sharex=True, sharey=True,)
            for j in range(0, cols):
                axarr[j].imshow(list_imgs[j+(rows - auxr)*cols][1], cmap='gray', vmin = 0.0, vmax= 1.0)
                axarr[j].title.set_text(list_imgs[j+(rows - auxr)*cols][0])
            auxr-=1
            plt.show()

        lastRowCols = len(list_imgs)-cols*rows
        
        if lastRowCols!=0 :
            _, axarr = plt.subplots(ncols=lastRowCols, figsize=(50,50), sharex=True, sharey=True)
            for i in range(0, lastRowCols):
                axarr[i].imshow(list_imgs[(cols*rows)+i][1], cmap='gray', vmin = 0.0, vmax= 1.0)
                axarr[i].title.set_text(list_imgs[(cols*rows)+i][0])

            plt.show()
    
    @staticmethod
    def get_gauss_2d_kernel(sigma, mu, rX, rY):
        x, y = np.meshgrid(np.linspace(-1,1,rX), np.linspace(-1,1,rY))
        d = np.sqrt(x*x+y*y)
        return np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

    @staticmethod
    def gkern(kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        return kernel_raw/kernel_raw.sum()

    @staticmethod
    def getRandomBinary():
        img_pre = np.random.randint(2, size=(20,20))
        return np.where(img_pre>0.2, 1.0, 0)
    
    @staticmethod
    def getRoberts(inImage):
        roberts_cross_v = np.array( [[ 0, 0, 0 ],
                                    [ 0, 1, 0 ],
                                    [ 0, 0,-1 ]] )
        roberts_cross_h = np.array( [[ 0, 0, 0 ],
                                    [ 0, 0, 1 ],
                                    [ 0,-1, 0 ]] )

        vertical = ndimage.convolve( inImage, roberts_cross_v )
        horizontal = ndimage.convolve( inImage, roberts_cross_h )

        return [vertical, horizontal]
        



