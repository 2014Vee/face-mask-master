#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : 2014Vee

import os
import numpy
from PIL import Image, ImageDraw
import cv2
import imghdr

# img = cv2.imread("./face/7.jpg")
filePath = "./face/face_00000481.jpg"
img1 = imghdr.what(filePath)

print(img1)

print(type(img1))

# 对于imread无法读取中文路径的解决办法
print(os.path.exists(filePath))
img = cv2.imdecode(numpy.fromfile(filePath,dtype=numpy.uint8),-1)
print("img:", img)
print("shape of img", img.shape)
print("len(img)", len(img))

#定义分类器（人眼识别）
classifier_eye = cv2.CascadeClassifier("./haarcascade_eye.xml")

#检测器识别眼睛
# 检测器：detectMultiScale参数（图像，每次缩小图像的比例，匹配成功所需要的周围矩形框的数目，检测的类型，匹配物体的大小范围）
faceRects_eye = classifier_eye.detectMultiScale(img, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, (20, 20))
print("faceRects", len(faceRects_eye))

#检测到眼睛后循环
if len(faceRects_eye) > 0:
    # 定义一个列表存放两只眼睛坐标
    eye_tag = []
    for faceRect_eye in faceRects_eye:
        x1, y1, w1, h2 = faceRect_eye
        # 画出眼睛区域
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1) + int(w1), int(y1) + int(h2)), (0, 255, 0), 2, 0)
        # 定义a变量获取眼睛坐标
        a = (y1, y1 + h2, x1, x1 + w1)
        # 通过append存入数组a中
        eye_tag.append(a)

    cv2.imwrite("./face/h6.jpg", img)

    # 存放为ndarray数组类型，输入内容为[[x1 y1 x1+w y1+h][x1 y1 x1+w y1+h]...],后面会获取多维数组的下标来替换数值
    n_eyetag = numpy.array(eye_tag)
# cv2.imshow('eye', img)
# cv2.waitKey(0)
