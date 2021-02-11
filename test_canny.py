from functions import *
from utils import *

def test_canny(url_i):
    img = utils.Utils.image_gray(url_i+'/'+'image.png')
    img = img/250
    res1 = functions.edgeCanny(img, 3, 0.002, 0.9)
    utils.Utils.img_save(url_out+'/'+ "canny1.png", res1)
    res2 = functions.edgeCanny(img, 3, 0.002, 0.002)
    utils.Utils.img_save(url_out+'/'+ "canny2.png", res2)
    imgs = [('original', img),('canny1', res1), ('canny2', res2)]
    utils.Utils.plot_test_compare_n(imgs, 3)