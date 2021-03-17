from skimage.measure import compare_ssim,compare_psnr
import numpy as np
import cv2
import skimage
from skimage.color import rgb2gray

# def get_file(num):
#     file = open("places365_val.txt", "r")
#     lines = file.read().split('\n')
#     ress = [line.split(' ')[:2] for line in lines]
#     ress = [res[0] for res in ress if res[-1] == str(num)]
#     return ress


def test_quality():
    path = "/data/generative_inpainting/results_place2/"
    #    file_names = get_file(81)
    ssim_scores = []
    psnr_scores = []
    l1_losses = []
    file_names = [("00" + str(i + 1))[-3:] + "_im" for i in range(100)]

    for name in file_names:
        fake_path = path + name + "_fake_B" + ".png"
        real_path = path + name + "_real_B" + ".png"
        # imageA = cv2.imread(fake_path)
        # imageB = cv2.imread(real_path)

        imageA = (cv2.imread(fake_path) / 255.0).astype(np.float32)
        imageB = (cv2.imread(real_path) / 255.0).astype(np.float32)

        imageA = rgb2gray(imageA)
        imageB = rgb2gray(imageB)

        rec_img = np.array(imageB)

        # rec_img[64:192, 64:192, :] = imageA[64:192, 64:192, :]

        rec_img[64:192, 64:192] = imageA[64:192, 64:192]
        imageA = rec_img

        psnr_scores.append(compare_psnr(imageA, imageB, data_range=1))
        ssim_scores.append(compare_ssim(imageA, imageB, data_range=1))
        l1_losses.append(compare_l1(imageA, imageB))

        # (score, diff) = compare_ssim(imageA, imageB, multichannel=True, full=True)
        # diff = (diff * 255).astype("uint8")
        # ssim_scores.append(score)
        # psnr_scores.append(psnr1(imageA, imageB))
        # l1_losses.append(L1loss(imageA, imageB))

    print("SSIM score is:", sum(ssim_scores) / len(ssim_scores))
    print("PSNR score is:", sum(psnr_scores) / len(psnr_scores))
    print("L1 Losses score is:", sum(l1_losses) / len(l1_losses))


def psnr1(img1, img2):
    return skimage.measure.compare_psnr(img1, img2)


def L1loss(img1, img2):
    return np.mean(np.abs(img1/255 - img2/255))

def compare_l1(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.mean(np.abs(img_true- img_test))

test_quality()