from functions import *
from utils import *


def test_fill(url_i, url_out):
    img = utils.Utils.image_gray(url_i+'/'+'image 2.png')
    img = img / 250
    res1 = functions.fill(img, [(24,24)], [[1,1,1],[1,1,1],[1,1,1]] )
    res2 = functions.fill(img, [(21,44), (5,5)])
    utils.Utils.img_save(url_out+'/'+ "fill1.png", res1)
    utils.Utils.img_save(url_out+'/'+ "fill2.png", res2)
    utils.Utils.plot_test_compare(img, res1, res2)