from vap1lib import functions, utils
import skimage as ski
from astropy.convolution import Gaussian2DKernel
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
import numpy as np
import cv2
from skimage import feature
from skimage import data
from skimage import color
from skimage import img_as_float


class Test:

    def __init__(self, url_i='~/Documentos/A_Cuarto/VA/P1', img_i='lena.png'):
        self.url  = url_i
        self.img_pre = utils.Utils.image_gray(self.url+'/'+img_i)

    def get_image(self):
        return self.img_pre

    def load_simple_binary(self, imput = None):
        
        if imput is None:
            self.img_pre = np.array([
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            [0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
            [0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0],
            [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],
            [0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],
            [0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0],
            [0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0],
            [0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0],
            [0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0],
            [0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0],
            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]])
        else:
            self.img_pre = imput    

    def load_new_image(self, nImage = ''):
        if not nImage:
            self.img_pre = utils.Utils.getRandomBinary()
        else:
            self.img_pre = utils.Utils.image_gray(self.url+"/"+nImage)

    def test_adjustIntensity(self):
        my_image = functions.adjustIntensity(self.img_pre, inRange = [0.2, 0.8], outRange=[0.4,0.6])
        alg_image = ski.exposure.rescale_intensity(self.img_pre, in_range=(0.2, 0.8), out_range=(0.4,0.6))
        utils.Utils.plot_test_compare(self.img_pre, alg_image, my_image)

    def test_equalizeIntensity(self):
        my_image = functions.equalizeIntensity(self.img_pre)
        alg_image = ski.exposure.equalize_hist(self.img_pre)
        utils.Utils.plot_test_compare(self.img_pre, alg_image, my_image)

    def test_filterImage(self):
        my_filtered = functions.filterImage(self.img_pre, Gaussian2DKernel(4).array)
        alg_filtered = gaussian_filter(self.img_pre, sigma=4)
        utils.Utils.plot_test_compare(self.img_pre, my_filtered, alg_filtered)

    def test_gaussianFilter(self):
        my_filtered = functions.gaussianFilter(self.img_pre, 0.6)
        alg_filtered = gaussian_filter(self.img_pre, 0.2)
        utils.Utils.plot_test_compare(self.img_pre, my_filtered, alg_filtered)
    
    def test_medianFilter(self):
        my_filtered = functions.medianFilter(self.img_pre, 3)
        alg_filtered = medfilt(self.img_pre, 3)
        utils.Utils.plot_test_compare(self.img_pre, my_filtered, alg_filtered)
    
    def test_highBoost(self):
        my_gauss_1 = functions.highBoost(self.img_pre, 1, 'gaussian', 2)
        my_gauss_2 = functions.highBoost(self.img_pre, 2, 'gaussian', 2)
        my_gauss_3 = functions.highBoost(self.img_pre, 3, 'gaussian', 2)

        my_median_1 = functions.highBoost(self.img_pre, 1, 'median', 3)
        my_median_2 = functions.highBoost(self.img_pre, 2, 'median', 3)
        my_median_3 = functions.highBoost(self.img_pre, 3, 'median', 3)

        imgs = [
            ('original',self.img_pre),('my_gaussian_1',my_gauss_1), ('my_gaussian_2',my_gauss_2), ('my_gaussian_3',my_gauss_3), 
            ('original',self.img_pre), ('my_median_1',my_median_1),('my_median_2',my_median_2),('my_median_3',my_median_3)
        ]
        utils.Utils.plot_test_compare_n(imgs, 4)

    def test_erode(self):
        dilated1 = functions.erode(self.img_pre,np.array([[1., 1., 1.]]),[0,0])
        dilated2 = functions.erode(self.img_pre,np.array([[1., 1., 1.]]),[0,1])
        dilated3 = functions.erode(self.img_pre,np.array([[1., 1., 1.]]),[0,2])
        dilated4 = functions.erode(self.img_pre,np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]))
        cv_dilate = cv2.erode(self.img_pre,np.array([[1., 1., 1.]]))
        cv_dilate2 = cv2.erode(self.img_pre, np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]))


        imgs=[('original', self.img_pre),('erode', dilated1),('cv.erode', cv_dilate),
              ('original', self.img_pre),('erode1', dilated2),('cv.erode1', cv_dilate),
              ('original', self.img_pre),('erode2', dilated3),('cv.erode2', cv_dilate),
              ('original', self.img_pre),('erode3', dilated4),('cv.erode3', cv_dilate2)
            ]
        utils.Utils.plot_test_compare_n(imgs,3)

    def test_gray_scale_erode(self):
        dilated1 = functions.gray_scale_erode(self.img_pre,np.array([[0.1, 0.2, 0.1]]),[0,0])
        dilated2 = functions.gray_scale_erode(self.img_pre,np.array([[0.1, 0.2, 0.1]]),[0,1])
        dilated3 = functions.gray_scale_erode(self.img_pre,np.array([[0.1, 0.2, 0.1]]),[0,2])
        dilated4 = functions.gray_scale_erode(self.img_pre,np.array([[0.1, 0.2, 0.1], [0.2, 0.2, 0.2], [0.1, 0.2, 0.1]]))
        cv_dilate = cv2.erode(self.img_pre,np.array([[0.1, 0.2, 0.1]]))
        cv_dilate2 = cv2.erode(self.img_pre, np.array([[0.1, 0.2, 0.1], [0.2, 0.2, 0.2], [0.1, 0.2, 0.1]]))


        imgs=[('original', self.img_pre),('erode', dilated1),('cv.erode', cv_dilate),
              ('original', self.img_pre),('erode1', dilated2),('cv.erode1', cv_dilate),
              ('original', self.img_pre),('erode2', dilated3),('cv.erode2', cv_dilate),
              ('original', self.img_pre),('erode3', dilated4),('cv.erode3', cv_dilate2)
            ]
        utils.Utils.plot_test_compare_n(imgs,3)

    def test_gray_scale_dilate(self):
        dilated1 = functions.gray_scale_dilate(self.img_pre,np.array([[0.1, 0.2, 0.1]]),[0,0])
        dilated2 = functions.gray_scale_dilate(self.img_pre,np.array([[0.1, 0.2, 0.1]]),[0,1])
        dilated3 = functions.gray_scale_dilate(self.img_pre,np.array([[0.1, 0.2, 0.1]]),[0,2])
        dilated4 = functions.gray_scale_dilate(self.img_pre,np.array([[0.1, 0.2, 0.1], [0.2, 0.2, 0.2], [0.1, 0.2, 0.1]]))
        cv_dilate = cv2.erode(self.img_pre,np.array([[0.1, 0.2, 0.1]]))
        cv_dilate2 = cv2.erode(self.img_pre, np.array([[0.1, 0.2, 0.1], [0.2, 0.2, 0.2], [0.1, 0.2, 0.1]]))

        cv_dilate = cv2.dilate(self.img_pre, np.array([[1., 1., 1.]]))
        cv_dilate2 = cv2.dilate(self.img_pre, np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]))

        imgs=[('original', self.img_pre),('dilated1', dilated1),('cv.dilate', cv_dilate),
              ('original', self.img_pre),('dilated2', dilated2),('c.dilate2',cv_dilate),
              ('original', self.img_pre),('dilated3', dilated3),('c.dilate3',cv_dilate),
              ('original', self.img_pre),('dilated4', dilated4),('c.dilate', cv_dilate2)]
        utils.Utils.plot_test_compare_n(imgs,3)

    def test_dilate(self):
        dilated1 = functions.dilate(self.img_pre,np.array([[1., 1., 1.]]),[0,0])
        dilated2 = functions.dilate(self.img_pre,np.array([[1., 1., 1.]]),[0,1])
        dilated3 = functions.dilate(self.img_pre,np.array([[1., 1., 1.]]),[0,2])
        dilated4 = functions.dilate(self.img_pre,np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]))

        cv_dilate = cv2.dilate(self.img_pre, np.array([[1., 1., 1.]]))
        cv_dilate2 = cv2.dilate(self.img_pre, np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]))

        imgs=[('original', self.img_pre),('dilated1', dilated1),('cv.dilate', cv_dilate),
              ('original', self.img_pre),('dilated2', dilated2),('c.dilate2',cv_dilate),
              ('original', self.img_pre),('dilated3', dilated3),('c.dilate3',cv_dilate),
              ('original', self.img_pre),('dilated4', dilated4),('c.dilate', cv_dilate2)]
        utils.Utils.plot_test_compare_n(imgs,3)

    def test_fill(self):
        alg_c = 1.0 - self.img_pre
        (img_x, img_y) = self.img_pre.shape
        fill_alg1 = functions.fill(self.img_pre,[(int(img_x/2),int(img_y/2))])
        fill_alg2 = functions.fill(self.img_pre,[(7,3)])
        fill_alg3 = functions.fill(self.img_pre,[(4,6), (7,3)]) 
        imgs = [('original', self.img_pre), ('example', alg_c), ('fill_1', fill_alg1), ('fill_2', fill_alg2), ('fill_3', fill_alg3)]
        utils.Utils.plot_test_compare_n(imgs,3)

    def test_gradientImage(self):
        [my_robert_x, my_robert_y] = functions.gradientImage(self.img_pre, 'Roberts')
        [my_cd_x, my_cd_y] = functions.gradientImage(self.img_pre, 'CentralDiff')
        [my_prewitt_x, my_prewitt_y] = functions.gradientImage(self.img_pre, 'Prewitt')
        [my_sobel_x, my_sobel_y] = functions.gradientImage(self.img_pre, 'Sobel')

        imgs = [
        ('original',self.img_pre), ("Roberts Gx", my_robert_x), ("Roberts Gy", my_robert_y),
        ('original',self.img_pre), ("CentralDiff Gx", my_cd_x), ("CentralDiff Gy", my_cd_y),
        ('original',self.img_pre), ("Prewitt Gx", my_prewitt_x), ("Prewitt Gy", my_prewitt_y),
        ('original',self.img_pre), ("Sobel Gx", my_sobel_x), ("Sobel Gy", my_sobel_y),        
        ]
        utils.Utils.plot_test_compare_n(imgs, 3)

    def test_edgeCanny(self):
        out = functions.edgeCanny(self.img_pre, 0.1, 0.1, 0.19)
        alg = feature.canny(self.img_pre, sigma=1.2)
        utils.Utils.plot_test_compare( self.img_pre, out, alg, False)

    def test_cornerHarrys(self):
        corners = functions.cornerHarrys(self.img_pre, 0.2, 0.2, 5.0)

        
        grayscale_image = img_as_float(corners)
        aux = color.gray2rgb(grayscale_image)


        utils.Utils.plot_gray_color(self.img_pre,aux)


