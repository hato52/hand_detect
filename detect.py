## detect.py
## 画像から手の領域を検出する

import os, sys, time
import cv2
import numpy as np

## HOGDescriptorのインスタンス化
win_size = (100, 100)       # 検出窓のサイズ
block_size = (16, 16)       # ブロックのサイズ
block_stride = (4, 4)       # ストライド幅
cell_size = (8, 8)          # セルサイズ
nbins = 9

hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

## SVMの読み込み、サポートベクトルの取得
svm = cv2.ml.SVM_load("hand_edge_svm.xml")
primal = svm.getSupportVectors()
hog.setSVMDetector(np.array(primal))

## 画像の読み込み
argv = sys.argv
img = cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
result_img = cv2.imread(argv[1], cv2.IMREAD_COLOR)

## エッジ画像に変換
img = cv2.GaussianBlur(img, (5,5), 3, 3)
img = cv2.Canny(img, 50, 100)

## 検出
start = time.time() # 処理時間の計算
found, w = hog.detectMultiScale(img, winStride=(16, 16), padding=(4, 4), scale=1.5, finalThreshold=5)
#found, w = hog.detectMultiScale(img)
for x, y, w, h in found:
    pad_w, pad_h = int(0.1*w), int(0.1*h)
    cv2.rectangle(result_img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), 3)
    cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (255, 255, 255), 3)
    #cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 3)

## 結果の表示
elapsed = time.time() - start
print("elapsed time: " + str(elapsed))
cv2.imshow("result", result_img)
cv2.imshow("canny", img)

cv2.waitKey(0)
cv2.destroyAllWindows()