# coding:utf-8
## optical_flow.py
## オプティカルフローのサンプル

from __future__ import print_function
import sys
import numpy as np
import pyrealsense2 as rs
import cv2

## パイプラインの用意
pipeline = rs.pipeline()

## キャプチャするフレームの設定
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

## ストリーミングの開始
profile = pipeline.start(config)

saved_frame = None
flag = True

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

        ## キャプチャ画像をnumpy配列に変換
        color_frame = np.asanyarray(color.get_data())
        depth_frame = np.asanyarray(depth.get_data())

        gray_img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

        if flag == True:
            hsv = np.zeros_like(color_frame)
            hsv[...,1] = 255

        ## オプティカルフローの計算
        if flag == False:
            flow = cv2.calcOpticalFlowFarneback(saved_frame, gray_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

            cv2.imshow("flow", rgb)

        ## キャプチャしたフレームを一時的に保持
        saved_frame = gray_img
        flag = False

        ## OpenCVで表示
        cv2.imshow("color", color_frame)
        key = cv2.waitKey(1)
        if (key == 113): break

finally:
    pipeline.stop()