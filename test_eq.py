from functions import *
from utils import * 


def test_equalize(url_i,url_out):
    eq = utils.Utils.image_gray(url_i+'/'+'eq.png')
    eq = eq/250

    res = functions.equalizeIntensity(eq, 50)
    utils.Utils.img_save(url_out+'/'+ "eqRes.png", res)
    utils.Utils.plot_test_compare(eq, res, res)