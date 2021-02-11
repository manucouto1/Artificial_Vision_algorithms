from functions import *
from utils import *


def test_filterImage(url_i):
    filter1 = utils.Utils.image_gray(url_i+'/'+'filter1.png')
    kernel = utils.Utils.image_gray(url_i+'/'+'kernel.png')
    filter1 = filter1/250
    kernel = kernel/250
    
    res = functions.filterImage(filter1, kernel)
    utils.Utils.img_save(url_out+'/'+ "filter.png", res)
    utils.Utils.plot_test_compare(filter1, res, res)