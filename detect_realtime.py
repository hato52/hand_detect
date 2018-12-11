#coding:utf-8
## detect_realtime.py
## RealSenseのカメラから取得した映像で検出を行うテスト

from __future__ import print_function
import numpy as np
import pyrealsense2 as rs
import cv2
import sys

## HOG-SVM分類器による手の検出を行う
def detectHand(img, hog, depth):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray_img = cv2.GaussianBlur(gray_img, (5,5), 3, 3)
    #gray_img = cv2.Canny(gray_img, 50, 100)

    # 検出
    hands, w = hog.detectMultiScale(gray_img, hitThreshold=-1, winStride=(16, 16), padding=(4, 4), scale=1.2, finalThreshold=5)
    
    for x, y, w, h in hands:
        pad_w, pad_h = int(0.1*w), int(0.1*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 200, 0), 3)

## 画像の背景除去
def removeBackground(color, depth, clipping):
    depth_frame_3d = np.dstack((depth, depth, depth))
    removed = np.where((depth_frame_3d > clipping) | (depth_frame_3d <= 0), 153, color)

    return removed

def main():
    ## パイプラインの用意
    pipeline = rs.pipeline()

    ## キャプチャするフレームの設定
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    ## 後処理用のフィルタ設定
    spa_filter = rs.spatial_filter()    # 画像の平滑化
    spa_filter.set_option(rs.option.holes_fill, 3)

    tem_filter = rs.temporal_filter()    # 時間的なノイズの低減
    tem_filter.set_option(rs.option.holes_fill, 1)

    ## ストリーミングの開始
    profile = pipeline.start(config)

    ## 除去を行う距離を算出
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    clipping_distance = 1.0 / depth_scale

    ## プリセットの適用
    depth_sensor.set_option(rs.option.visual_preset, 2)

    ## HOGDescriptorのインスタンス化
    win_size = (100, 100)       # 検出窓のサイズ
    block_size = (16, 16)       # ブロックのサイズ
    block_stride = (4, 4)       # ストライド幅
    cell_size = (8, 8)          # セルサイズ
    nbins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    ## SVM分類器にサポートベクトルをセット
    svm = cv2.ml.SVM_load("hand_svm.xml")
    primal = svm.getSupportVectors()
    hog.setSVMDetector(np.array(primal))

    try:
        while True:
            ## 新規フレームを待機
            frames = pipeline.wait_for_frames()

            ## デプスフレームをRGBフレームに合わせる
            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frames)

            ## 各種フレームを取得
            color = aligned_frames.get_color_frame()
            depth = aligned_frames.get_depth_frame()

            if not depth or not color: continue

            ## フィルタを適用
            filtered_depth = depth
            #filtered_depth = spa_filter.process(filtered_depth)
            #filtered_depth = tem_filter.process(filtered_depth)

            ## キャプチャ画像をnumpy配列に変換
            color_frame = np.asanyarray(color.get_data())
            depth_frame = np.asanyarray(filtered_depth.get_data())

            ## 取得した画像の背景除去
            removed = color_frame
            #removed = removeBackground(color_frame, depth_frame, clipping_distance)

            ## 手の検出
            detectHand(removed, hog, depth)

            ## OpenCVで表示
            cv2.imshow("window", removed)
            key = cv2.waitKey(1)
            if (key == 113): break

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()