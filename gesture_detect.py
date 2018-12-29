# coding:utf-8
## hand_tracking.py
## ジェスチャの検出を行う

from __future__ import print_function
import sys
import numpy as np
import pyrealsense2 as rs
import cv2
import win32pipe, win32file

## 取得する画像サイズ
IMG_WIDTH = 640
IMG_HEIGHT = 480

## 背景除去を行う距離
CLIPPING_DISTANCE_IN_METERS = 1.0

## モーション判定のために保持するフレーム数
SAVE_FRAME_SIZE = 5

## モーション検出の閾値
THRESHOLD_X = 10
THRESHOLD_Y = 10
THRESHOLD_Z = 0.015

## モーション
NO_MOTION = 0
PULL = 1
PUSH = 2
LEFT = 3
RIGHT = 4
DOWN = 5
UP = 6

## モーション検出用のリスト
horizon_ary = []
vertical_ary = []
dist_ary = []

## カスケード分類器による手の検出を行う
# def detectHandInImage(image, cascade, depth):
#     # 手の検出
#     hands = cascade.detectMultiScale(image, 1.6, 3, 0, (30, 30))

#     nearest = {"x" : 0, "y" : 0, "w" : 0, "h" : 0}
#     nearest_dist = 100.0

#     if len(hands) == 0: return None, None, None

#     # 検出したオブジェクトの中で最も近いものを判定
#     for x, y, w, h in hands:
#         if depth:
#             tmp = round(depth.get_distance(x + (w / 2), y + (h / 2)), 3)
#             if nearest_dist > tmp:
#                 nearest_dist = tmp
#                 nearest["x"] = x
#                 nearest["y"] = y
#                 nearest["w"] = w
#                 nearest["h"] = h
                      
#     # 一番近いものを手として検出
#     if nearest_dist <= 0.8:
#         cv2.rectangle(image, (nearest["x"], nearest["y"]), (nearest["x"] + nearest["w"], nearest["y"] + nearest["h"]), (0, 200, 0), 3)

#     x = nearest["x"] + (nearest["w"] / 2)
#     y = nearest["y"] + (nearest["h"] / 2)

#     return x, y, nearest_dist


## HOG-SVM分類器による手の検出を行う
def detectHand(img, hog, depth):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 検出
    hands, w = hog.detectMultiScale(gray_img, hitThreshold=-1, winStride=(16, 16), padding=(4, 4), scale=1.2, finalThreshold=5)

    nearest = {"x" : 0, "y" : 0, "w" : 0, "h" : 0}
    nearest_dist = 100.0

    if len(hands) == 0: return None, None, None

    ## 検出したオブジェクトの中で最も近いものを判定
    for x, y, w, h in hands:
        if depth:
            tmp = round(depth.get_distance(x + (w / 2), y + (h / 2)), 3)
            if nearest_dist > tmp:
                nearest_dist = tmp
                nearest["x"] = x
                nearest["y"] = y
                nearest["w"] = w
                nearest["h"] = h

    if nearest_dist <= 0.8:
        pad_w, pad_h = int(0.1*nearest["w"]), int(0.1*nearest["h"])
        cv2.rectangle(img, (nearest["x"]+pad_w, nearest["y"]+pad_h), (nearest["x"]+nearest["w"]-pad_w, nearest["y"]+nearest["h"]-pad_h), (0, 200, 0), 3)

    x = nearest["x"] + (nearest["w"] / 2)
    y = nearest["y"] + (nearest["h"] / 2)

    return x, y, nearest_dist


## 画像の背景除去
def removeBackground(color, depth, clipping):
    depth_frame_3d = np.dstack((depth, depth, depth))
    removed = np.where((depth_frame_3d > clipping) | (depth_frame_3d <= 0), 153, color)

    return removed


## X軸方向のモーションを検出する
def detectHandMoved_X(x):
    global horizon_ary

    # リストに手のx座標をプッシュ
    horizon_ary.append(x)

    # 5フレーム分溜まっていなければ処理は終了
    if len(horizon_ary) <= SAVE_FRAME_SIZE: return 0

    # 一番古い要素を削除
    horizon_ary.pop(0)

    # フレームごとの変化量を取得
    motion_flg = 0
    for i in range(len(horizon_ary) - 1):
        if horizon_ary[i + 1] - horizon_ary[i] > THRESHOLD_X:
            motion_flg += 1
        elif horizon_ary[i + 1] - horizon_ary[i] < -THRESHOLD_X:
            motion_flg -= 1
        else:
            return 0

    # 全て一定以上の変化量であればモーションあり
    # 保持フレームを空にする
    del horizon_ary[:]

    if motion_flg == SAVE_FRAME_SIZE - 1:
        print("RIGHT")
        return RIGHT
    elif motion_flg == -(SAVE_FRAME_SIZE - 1):
        print("LEFT")
        return LEFT

    return 0


## Y軸方向のモーションを検出する
def detectHandMoved_Y(y):
    global vertical_ary

    # キューに手のx座標をプッシュ
    vertical_ary.append(y)

    #5フレーム分溜まっていなければ処理は終了
    if len(vertical_ary) <= SAVE_FRAME_SIZE: return 0

    # 一番古い要素を削除
    vertical_ary.pop(0)

    # フレームごとの変化量を取得
    motion_flg = 0
    for i in range(len(vertical_ary) - 1):
        if vertical_ary[i + 1] - vertical_ary[i] > THRESHOLD_Y:
            motion_flg += 1
        elif vertical_ary[i + 1] - vertical_ary[i] < -THRESHOLD_Y:
            motion_flg -= 1
        else:
            return 0

    # 全て一定以上の変化量であればモーションあり
    # 保持フレームを空にする
    del vertical_ary[:]

    if motion_flg == SAVE_FRAME_SIZE - 1:
        print("DOWN")
        return DOWN
    elif motion_flg == -(SAVE_FRAME_SIZE - 1):
        print("UP")
        return UP

    return 0


## Z軸方向のモーションを検出する
def detectHandMoved_Z(z):
    global dist_ary

    # キューに手のx座標をプッシュ
    dist_ary.append(z)

    # 5フレーム分溜まっていなければ処理は終了
    if len(dist_ary) <= SAVE_FRAME_SIZE: return 0

    # 一番古い要素を削除
    dist_ary.pop(0)

    # フレームごとの変化量を取得
    motion_flg = 0
    for i in range(len(dist_ary) - 1):
        if dist_ary[i + 1] - dist_ary[i] > THRESHOLD_Z:
            motion_flg += 1
        elif dist_ary[i + 1] - dist_ary[i] < -THRESHOLD_Z:
            motion_flg -= 1
        else:
            return NO_MOTION

    # 全て一定以上の変化量であればモーションあり
    # 保持フレームを空にする
    del dist_ary[:]

    if motion_flg == SAVE_FRAME_SIZE - 1:
        print("PULL")
        return PULL
    elif motion_flg == -(SAVE_FRAME_SIZE - 1):
        print("PUSH")
        return PUSH

    return NO_MOTION


## 名前付きパイプによるプロセス間通信を行う
def sendMessageToServer(file_handle, message):
    if (message == 0): return

    if message == PULL:
        send_mes = "pull"
    elif message == PUSH:
        send_mes = "push"
    elif message == LEFT:
        send_mes = "left"
    elif message == RIGHT:
        send_mes = "right"
    elif message == DOWN:
        send_mes = "down"
    elif message == UP:
        send_mes = "up"
    else:
        send_mes = None

    try:
        win32file.WriteFile(file_handle, send_mes)
    except Exception:
        print("Failed to send message")


def main():

    # カスケードファイルの読み込み
    # cascade_file = "aGest.xml"
    # cascade = cv2.CascadeClassifier()
    # cascade.load(cascade_file)

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

    ## プロセス間通信用のパイプへ接続する
    pipe_handle = None
    try:
        pipe_handle = win32file.CreateFile(
            '\\\\.\\pipe\\mypipe',
            win32file.GENERIC_WRITE,
            0, 
            None, 
            win32file.OPEN_EXISTING, 
            win32file.FILE_ATTRIBUTE_NORMAL,
            None)
    except Exception:
        print("Failed to connect to pipe")

    ## パイプラインの用意
    pipeline = rs.pipeline()

    ## キャプチャするフレームの設定
    config = rs.config()
    config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, IMG_WIDTH, IMG_HEIGHT, rs.format.z16, 30)

    ## ストリーミングの開始
    profile = pipeline.start(config)

    ## 後処理用のフィルタ設定
    # spa_filter = rs.spatial_filter()    # 画像の平滑化
    # spa_filter.set_option(rs.option.holes_fill, 3)

    # tem_filter = rs.temporal_filter()    # 時間的なノイズの低減
    # tem_filter.set_option(rs.option.holes_fill, 1)

    ## 除去を行う距離を算出
    # depth_sensor = profile.get_device().first_depth_sensor()
    # depth_scale = depth_sensor.get_depth_scale()
    # clipping_distance = CLIPPING_DISTANCE_IN_METERS / depth_scale

    ## プリセットの適用
    # depth_sensor.set_option(rs.option.visual_preset, 2)

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
            # filtered_depth = spa_filter.process(filtered_depth)
            # filtered_depth = tem_filter.process(filtered_depth)

            ## キャプチャ画像をnumpy配列に変換
            color_frame = np.asanyarray(color.get_data())
            depth_frame = np.asanyarray(filtered_depth.get_data())

            ## 取得した画像の背景除去
            removed = color_frame
            #removed = removeBackground(color_frame, depth_frame, clipping_distance)

            ## 手の検出
            #x, y, dist = detectHandInImage(color_frame, cascade, depth)
            x, y, dist = detectHand(removed, hog, depth)
 
            ## モーションの検出
            message = NO_MOTION
            if message == NO_MOTION and x != None:
                message = detectHandMoved_X(x)
            if message == NO_MOTION and y != None:
                message = detectHandMoved_Y(y)
            if message == NO_MOTION and dist != None and dist != 0.0:
                message = detectHandMoved_Z(dist)

            ## 結果を通知
            sendMessageToServer(pipe_handle, message)

            ## OpenCVで表示
            cv2.imshow("window", removed)
            key = cv2.waitKey(1)
            if (key == 113): break

    finally:
        pipeline.stop()
        win32file.CloseHandle(pipe_handle)

if __name__ == "__main__":
    main()