from functions import *
from utils import * 


def test_medianFilter(url_i,url_out):
    filter1 = utils.Utils.image_gray(url_i+'/'+'medianF.png')
    filter1 = filter1/250

    res1 = functions.medianFilter(filter1, 3)
    res2 = functions.medianFilter(filter1, 5)
    res3 = functions.medianFilter(filter1, 7)

    utils.Utils.img_save(url_out+'/'+ "Mfilter1.png", res1)
    utils.Utils.img_save(url_out+'/'+ "Mfilter2.png", res2)
    utils.Utils.img_save(url_out+'/'+ "Mfilter3.png", res3)

    utils.Utils.plot_test_compare(res1, res2, res3)