#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : 2014Vee


# !/usr/bin/env python
# coding=utf-8
import os
import numpy
from PIL import Image, ImageDraw
import cv2
#created by chenfenyu 2018.3.20


cap = cv2.VideoCapture(0)
#获取外接摄像头
eye = cv2.imread("/Users/funny/Downloads/img/eye.png")
#读取眼睛区域替换的图片
mouth = cv2.imread("/Users/funny/Downloads/img/mouth.png")
#读取嘴巴区域替换的图片
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5))
#获取摄像头返回的宽和高
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#确定保存视频的格式
video = cv2.VideoWriter("/Users/funny/Downloads/a.avi", fourcc, 5, size)
''' cv.VideoWriter参数（视频存放路径，视频存放格式，fps帧率，视频宽高）
    注意点1：OpenCV只支持avi的格式，而且生成的视频文件不能大于2GB，而且不能添加音频
    注意点2：若填写的文件名已存在，则该视频不会录制成功，但可正常使用
'''
print(cap.isOpened())
#检测是否摄像头正常打开:成功打开时，isOpened返回ture
classifier_face = cv2.CascadeClassifier("/Users/funny/Documents/haarcascade_frontalface_alt.xml")
#定义分类器（人脸识别）
classifier_eye = cv2.CascadeClassifier("/Users/funny/Documents/haarcascade_eye.xml")
#定义分类器（人眼识别）
classifier_mouth=cv2.CascadeClassifier("/Users/funny/Documents/haarcascade_mcs_mouth.xml")
#定义分类器（嘴巴识别）
while (True):
    #取得cap.isOpened()返回状态为True,即检测到人脸
    #img = cv2.imread("/Users/funny/Downloads/img/pp.png")
    ret, img = cap.read()
    '''第一个参数ret的值为True或False，代表有没有读到图片
       第二个参数是frame，是当前截取一帧的图片
    '''
    faceRects_face = classifier_face.detectMultiScale(img, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, (20, 20))
    #检测器：detectMultiScale参数（图像，每次缩小图像的比例，匹配成功所需要的周围矩形框的数目，检测的类型，匹配物体的大小范围）
    key = cv2.waitKey(1)
    #键盘等待
    if len(faceRects_face) > 0:
        #检测到人脸
        for faceRect_face in faceRects_face:
            x, y, w, h = faceRect_face
            #获取图像x起点,y起点,宽，高
            h1=int(float(h/1.5))
            #截取人脸区域高度的一半位置，以精确识别眼睛的位置
            intx=int(x)
            inty=int(y)
            intw=int(w)
            inth=int(h)
            #转换类型为int，方便之后图像截取
            my = int(float(y + 0.7 * h))
            #截取人脸区域下半部分左上角的y起点，以精确识别嘴巴的位置
            mh = int(0.4 * h)
            #截取人脸区域下半部分高度，以精确识别嘴巴的位置
            img_facehalf = img[inty:(inty+h1), intx:intx+intw]
            img_facehalf_bottom = img[my:(my + mh), intx:intx + intw]
            '''img获取坐标为，【y,y+h之间（竖）：x,x+w之间(横)范围内的数组】
               img_facehalf是截取人脸识别到区域上半部分
               img_facehalf_bottom是截取人脸识别到区域下半部分
            '''
            cv2.rectangle(img, (int(x), my), (int(x) + int(w), my + mh), (0, 255, 0), 2, 0)
            '''矩形画出区域 rectangle参数（图像，左顶点坐标(x,y)，右下顶点坐标（x+w,y+h），线条颜色，线条粗细）
                画出人脸识别下部分区域，方便定位
            '''
            faceRects_mouth = classifier_mouth.detectMultiScale(img_facehalf_bottom, 1.1, 1, cv2.CASCADE_SCALE_IMAGE, (5, 20))
            #嘴巴检测器
            if len(faceRects_mouth) > 0:
                for faceRect_mouth in faceRects_mouth:
                    xm1, ym1, wm1, hm2 = faceRect_mouth
                    cv2.rectangle(img_facehalf_bottom, (int(xm1), int(ym1)), (int(xm1) + int(wm1), int(ym1) + int(hm2)), (0,0, 255), 2, 0)
                    img_mx = cv2.resize(mouth, (wm1, hm2), interpolation=cv2.INTER_CUBIC)
                    #调整覆盖图片大小 resize参数（图像，检测到的（宽，高），缩放类型）
                    if key == ord('z'):
                    #检测当键盘输入z时，开始替换图片
                        img[my+ym1:(my+ym1+hm2), intx+xm1:(intx + xm1+wm1)] = img_mx
                        #将调整大小后的图片赋值给img
            cv2.rectangle(img, (int(x), int(y)), (int(x) + int(w), int(y) + int(h1)), (0, 255, 0), 2, 0)
            # 画出人脸识别上部分区域，方便定位
            faceRects_eye = classifier_eye.detectMultiScale(img_facehalf, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, (20, 20))
            #检测器识别眼睛
            if len(faceRects_eye) > 0:
                #检测到眼睛后循环
                eye_tag = []
                #定义一个列表存放两只眼睛坐标
                for faceRect_eye in faceRects_eye:
                    x1, y1, w1, h2 = faceRect_eye
                    cv2.rectangle(img_facehalf, (int(x1), int(y1)), (int(x1) + int(w1), int(y1) + int(h2)), (0, 255, 0), 2, 0)
                    #画出眼睛区域
                    a = ((inty+y1),(inty+y1 + h2), (intx+x1),(intx+x1 + w1))
                    #定义a变量获取眼睛坐标，现在img顶点位置已经改变，需要加上intx和inty的值才可以
                    eye_tag.append(a)
                    #通过append存入数组a中
                n_eyetag = numpy.array(eye_tag)
                #存放为ndarray数组类型，输入内容为[[x1 y1 x1+w y1+h][x1 y1 x1+w y1+h]...],后面会获取多维数组的下标来替换数值

                if len(faceRects_eye)==2:
                    #眼睛识别到两个时，同时替换图片
                    img_ex=cv2.resize(eye,(n_eyetag[0,1]-n_eyetag[0,0], n_eyetag[0,3]-n_eyetag[0,2]),interpolation=cv2.INTER_CUBIC)
                    img_ex1 = cv2.resize(eye, (n_eyetag[1, 1] - n_eyetag[1, 0], n_eyetag[1, 3] - n_eyetag[1, 2]), interpolation=cv2.INTER_CUBIC)
                    if key == ord('p'):
                    #检测到键盘输入p时，进行替换
                        img[n_eyetag[0,0]:n_eyetag[0,1],n_eyetag[0,2]:n_eyetag[0,3]]=img_ex
                        img[n_eyetag[1, 0]:n_eyetag[1, 1], n_eyetag[1, 2]:n_eyetag[1, 3]] = img_ex1
                if len(faceRects_eye)==1:
                    # 眼睛识别到一个时，替换图片
                    img_ex = cv2.resize(eye, (n_eyetag[0, 1] - n_eyetag[0, 0], n_eyetag[0, 3] - n_eyetag[0, 2]), interpolation=cv2.INTER_CUBIC)
                    if key == ord('p'):
                        img[n_eyetag[0, 0]:n_eyetag[0, 1], n_eyetag[0, 2]:n_eyetag[0, 3]] = img_ex

            video.write(img)
    cv2.imshow('video', img)
    #显示图片，标题名字为video
    cv2.resizeWindow('video',1280,720)
    #调整窗口大小video为1280*720
    if key == ord('q'):
    #检测到键盘输入q，退出循环
        break

video.release()
#不再录制视频
cap.release()
#释放摄像头
cv2.destroyAllWindows()
#关闭所有窗口显示
