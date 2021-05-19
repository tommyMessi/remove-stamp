# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
yz_img_root = r'\all_images_org'
mask_root = r'\img_mask_aug'
gt_root = r'\img_gt_aug'

img_aug = r'\img_aug'
def vertical_grad(src,color_start,color_end):
    h = src.shape[0]

    # print type(src)

    # 创建一幅与原图片一样大小的透明图片
    grad_img = np.ndarray(src.shape,dtype=np.uint8)

    # opencv 默认采用 BGR 格式而非 RGB 格式
    g_b = float(color_end[0] - color_start[0]) / h
    g_g = float(color_end[1] - color_start[1]) / h
    g_r = float(color_end[2] - color_start[2]) / h

    for i in range(h):
        for j in range(src.shape[1]):

                grad_img[i,j,0] = color_start[0] + i * g_b
                grad_img[i,j,1] = color_start[1] + i * g_g
                grad_img[i,j,2] = color_start[2] + i * g_r

    return grad_img

def remove_red_seal(input_img):
    blue_c, green_c, red_c = cv2.split(input_img)

    thresh, ret = cv2.threshold(red_c, 100, 255, cv2.THRESH_OTSU)
    thresh, ret_f = cv2.threshold(ret, 100, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('1', ret_f)
    # cv2.imshow('r', ret)
    # cv2.waitKey(0)
    thresh1, ret1 = cv2.threshold(blue_c, 100, 255, cv2.THRESH_OTSU)
    # cv2.imshow('b', ret1)
    # cv2.waitKey(0)
    result = ret_f+ret1
    # cv2.imshow('c', result)
    # cv2.waitKey(0)

    result_img_gray = np.expand_dims(green_c, axis=2)
    result_img_gray = np.concatenate((result_img_gray, result_img_gray, result_img_gray), axis=-1)

    aug_img = vertical_grad(input_img,(30,255,0 ),(30,255,4))
    blend = cv2.addWeighted(input_img,1.0,aug_img,0.6,0.0)
    # 加一个蒙版
    # result_img_inv = np.concatenate((red_c, blue_c, green_c), axis=2)

    return ret, result, result_img_gray, blend


yz_name_list = os.listdir(yz_img_root)
for yz_name in yz_name_list:
    print(yz_name)
    yz_path = os.path.join(yz_img_root, yz_name)
    yz_name_b = yz_name.replace('.png', '_black.png')
    yz_name_v = yz_name.replace('.png', '_inv.png')
    yz_path_b = os.path.join(img_aug, yz_name_b)
    gt_path_b = os.path.join(gt_root, yz_name_b)
    mask_path_b = os.path.join(mask_root, yz_name_b)

    yz_path_v = os.path.join(yz_img_root, yz_name_v)
    gt_path_v = os.path.join(gt_root, yz_name_v)

    mask_path = os.path.join(mask_root, yz_name)
    image = cv2.imread(yz_path)
    image_aug = cv2.imread(yz_path, 0)
    image = cv2.resize(image, (512,512))
    gt, mask, result_img_gray, result_img_inv = remove_red_seal(image)
    # cv2.imshow('gray', result_img_gray)
    # cv2.imshow('inv', result_img_inv)
    # cv2.waitKey(0)

    cv2.imwrite(yz_path_b, result_img_gray)
    cv2.imwrite(gt_path_b, gt)
    cv2.imwrite(mask_path_b, mask)



