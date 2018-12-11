# coding:utf-8
## negative.py
## ネガティブデータセットの作成
## 1つの画像からランダムに10個切り出し、任意の個数のネガティブ画像を作成する

from __future__ import print_function
import os, sys, random
import cv2

dir_path = "C:\\Users\\User\LearningData\\negative_data\\"
negative_img_files = os.listdir(dir_path)

j = 0
for neg_img in negative_img_files:
    img = cv2.imread(dir_path + neg_img)

    for i in range(10):
        height, width = img.shape[:2]
        x = random.randint(0, height-110)
        y = random.randint(0, width-110)

        # ランダムに100x100の画像を切り抜く
        generated_img = img[x:x+100, y:y+100]
        gray_img = cv2.cvtColor(generated_img, cv2.COLOR_RGB2GRAY)
        
        # エッジ画像に変形
        img_gus = cv2.GaussianBlur(gray_img, (5,5), 3, 3)
        img_cny = cv2.Canny(img_gus, 50, 100)

        cv2.imwrite("negative_data\\negative_" + str(i + j) + ".png", img_cny)
        print("GENERATE! negative_" + str(i + j))

    j += 10