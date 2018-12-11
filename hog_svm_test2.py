## hog_svm_test2.py
## コマンドライン引数に与えられた画像が手かどうか判別

import os, sys
import cv2
import numpy as np

## 判別したい画像のパスをコマンドライン引数から取得
argv = sys.argv

## HOGDescriptorの設定値
win_size = (100, 100)       # 検出窓のサイズ
block_size = (16, 16)       # ブロックのサイズ
block_stride = (4, 4)       # ストライド幅
cell_size = (8, 8)          # セルサイズ
nbins = 9

## HOGDescriptorのインスタンス化
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

hog_test = []

## HOG特徴量の算出
#print("Calculating HOG feature...")

img = cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, win_size)
#print(img)
cv2.imshow("test", img)
hog_test.append(hog.compute(img))

hog_test = np.array(hog_test)

## SVMの読み込み
svm = cv2.ml.SVM_load("hand_svm.xml")

## テストデータにSVMを適用
result = svm.predict(hog_test)
print(result)


# if result == 0:
#     print("手だよ")
# elif result == 1:
#     print("手じゃないよ")

cv2.waitKey(0)
cv2.destroyAllWindows()


