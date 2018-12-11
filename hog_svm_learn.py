# coding:utf-8
## hog_svm_learn.py
## OpenCVを用いて画像のHOG特徴量を計算し、SVMを学習させる

from __future__ import print_function
import os, sys
import cv2
import numpy as np

## 画像データのパスを取得
pos_dir_path = "positive_data\\"
neg_dir_path = "negative_data\\"

pos_img_files = os.listdir(pos_dir_path)
neg_img_files = os.listdir(neg_dir_path)

## HOGDescriptorの設定値
win_size = (100, 100)       # 検出窓のサイズ
block_size = (16, 16)       # ブロックのサイズ
block_stride = (4, 4)       # ストライド幅
cell_size = (8, 8)          # セルサイズ
nbins = 9

## HOGDescriptorのインスタンス化
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

# 学習用データ
t = pos_img_files + neg_img_files
#t = np.array(t)
# ラベル 正解が0, 間違いが1
tmp = np.hstack(( np.zeros(len(pos_img_files)), np.ones(len(neg_img_files)) ))
l = tmp.tolist()
#l = np.array(l, dtype=int)

# ランダムに学習するようにシャッフル
zipped = list(zip(t, l))
np.random.shuffle(zipped)
train_data, labels = zip(*zipped)

hog_ary = []

## HOG特徴量の算出
print("Calculating HOG feature...")
for img_name in train_data:
    img = cv2.imread("images\\" + img_name, cv2.IMREAD_COLOR)
    hog_ary.append(hog.compute(img))

hog_ary = np.array(hog_ary)
labels = np.array(labels, dtype=int)

## SVMの作成
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(0.5)

## SVMの学習
svm.train(hog_ary, cv2.ml.ROW_SAMPLE, labels)
svm.save("hand_svm_2.xml")
print("SVM saved!")

## 作成されたSVMの確認(正解率1になればOK)
result = svm.predict(hog_ary)[1].ravel()

cnt = 0
for i in range(len(labels)):
    if result[i] == labels[i]:
        cnt += 1

print("accuracy: " + str(cnt / len(labels)))