from skimage.measure import compare_ssim
import numpy as np
import cv2
import skimage


def get_file(num):
    file = open("celebA_list_eval_partition.txt", "r")
    lines = file.read().split('\n')
    ress = [line.split(' ')[:2] for line in lines]
    ress = [res[0] for res in ress if res[-1] == str(num)]
    return ress


def test_quality():
    path = "results_celeba//test_8//images//"
    file_names = []
    ress_test = get_file(2)
    for res in ress_test:
        file_names.append(res[:6])
    ssim_scores = []
    psnr_scores = []
    l1_losses = []
    i = 0
    # file_names = [("00" + str(i + 1))[-3:] + "_im" for i in range(100)]

    for name in file_names:
        fake_path = path + name + "_fake_B" + ".png"
        real_path = path + name + "_real_B" + ".png"
        imageA = cv2.imread(fake_path)
        imageB = cv2.imread(real_path)
        rec_img = np.array(imageB)
        rec_img[64:192, 64:192, :] = imageA[64:192, 64:192, :]
        imageA = rec_img

        (score, diff) = compare_ssim(imageA, imageB, multichannel=True, full=True)
        diff = (diff * 255).astype("uint8")
        ssim_scores.append(score)
        psnr_scores.append(psnr1(imageA, imageB))
        l1_losses.append(L1loss(imageA, imageB))
        if i % 400 == 0 :
            print("num : ",i)
            print("SSIM score is:", ssim_scores[i])
            print("PSNR score is:", psnr_scores[i])
            print("L1 Losses score is:", l1_losses[i])
            print()
        i = i +1

    print("SSIM score is:", sum(ssim_scores) / len(ssim_scores))
    print("PSNR score is:", sum(psnr_scores) / len(psnr_scores))
    print("L1 Losses score is:", sum(l1_losses) / len(l1_losses))


def psnr1(img1, img2):
    return skimage.measure.compare_psnr(img1, img2)


def L1loss(img1, img2):
    return np.mean(np.abs(img1/255 - img2/255))


test_quality()