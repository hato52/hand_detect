# coding:utf-8
## positive.py
## ポジティブデータセットの作成
## リサイズしてエッジ画像にする

from __future__ import print_function
import os, sys
import cv2

dir_path = "C:\\Users\\User\LearningData\hand_data\\"
positive_img_files = os.listdir(dir_path)

i = 0
for pos_img in positive_img_files:
    img = cv2.imread(dir_path + pos_img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))

    # エッジ画像に変形
    img_gus = cv2.GaussianBlur(img, (5,5), 3, 3)
    img_cny = cv2.Canny(img_gus, 50, 100)
    
    cv2.imwrite("positive_data\positive_" + str(i) + ".png", img_cny)
    print("GENERATE! positive_" + str(i))
    i += 1