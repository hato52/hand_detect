# coding:utf-8
## hog_svm_test.py
## 作成したSVMの性能をテスト

from __future__ import print_function
import os, sys
import cv2
import numpy as np

## テストデータのパスを取得
pos_dir_path = "test_positive\\"
neg_dir_path = "test_negative\\"

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

# テスト用データ
test_data = pos_img_files + neg_img_files
# ラベル
tmp = np.hstack(( np.ones(len(pos_img_files)), np.zeros(len(neg_img_files)) ))
labels = tmp.tolist()


hog_test = []

## HOG特徴量の算出
print("Calculating HOG feature...")
for img_name in test_data:
    img = cv2.imread("test_images\\" + img_name, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, win_size)
    hog_test.append(hog.compute(img))

hog_test = np.array(hog_test)
labels = np.array(labels, dtype=int)

## SVMの読み込み
svm = cv2.ml.SVM_load("hand_edge_svm.xml")

## テストデータにSVMを適用
result = svm.predict(hog_test)[1].ravel()

cnt = 0
for i in range(len(labels)):
    if result[i] == labels[i]:
        cnt += 1

print("num of test data is " + str(len(labels)))
print("test accuracy: " + str(cnt / len(labels)))
