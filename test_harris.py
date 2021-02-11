from functions import *
from utils import * 
from skimage import img_as_float, color
import matplotlib.pyplot as plt


def plt_color(color_img, output):
    corners_float = img_as_float(color_img)
    corner_rgb = color.gray2rgb(corners_float)
    corners = corner_rgb * [1,0,0]

    _, axarr = plt.subplots(ncols=1, figsize=(50,50), sharex=True, sharey=True)
    axarr.imshow(corners, vmin = 0.0, vmax= 1.0)
    plt.savefig(output)
    plt.show()

def gray_plus_color(gray_img, color_img, output):
    corners_float = img_as_float(color_img)
    corner_rgb = color.gray2rgb(corners_float)
    corners = corner_rgb * [1,0,0]

    img_float = img_as_float(gray_img)
    img_rgb = color.gray2rgb(img_float)
    img = img_rgb

    _, axarr = plt.subplots(ncols=1, figsize=(50,50), sharex=True, sharey=True)
    axarr.imshow(img+corners, vmin = 0.0, vmax= 1.0)
    plt.savefig(output)
    plt.show()

def test_harris(url_i,url_out):
    harris = utils.Utils.image_gray(url_i+'/'+'harris.png')
    [harrisMap, corners] = functions.cornerHarrys(harris, 0.02, 0.02, 1000)
    grayscale_image = img_as_float(harrisMap)
    aux = color.gray2rgb(grayscale_image)
    gray_plus_color(harris,aux,'image_harrisMap.png')
    plt_color(aux,"harrisMap.png")

    grayscale_image = img_as_float(corners)
    aux = color.gray2rgb(grayscale_image)
    gray_plus_color(harris,aux,'image_corners.png')
    plt_color(aux,"corners.png")