#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : 2014Vee

import os
import numpy
from PIL import Image, ImageDraw
import cv2
import shutil
import imghdr
import random

def rename_data(file_path, shuffle = False):
    """
    本函数用于格式化命名文件夹下的图片名字，默认更改为jpg后缀
    提供随机混洗文件功能
    :param file_path: 输入需要格式化命名的文件夹
    :param shuffle:   是否需要对文件进行混洗，默认为False
    :return:
    """
    filelist = os.listdir(file_path)  # 列举图片
    i = 0
    needcount = len(filelist)
    # 随机混洗方法，利用random生成随机数
    randomlist = random.sample(range(1, needcount + 1), needcount)
    for item in filelist:
        total_num_file = len(filelist)  # 单个文件夹内图片的总数
        if item.endswith('.jpg'):
            src = os.path.join(os.path.abspath(file_path), item)  # 原图的地址
            # 新图的地址（这里可以把str(folder) + '_' + str(i) + '.jpg'改成你想改的名称）
            dst = os.path.join(os.path.abspath(file_path), 'face' + '_' + str(
                i if shuffle else randomlist[i]).zfill(8) + '.jpg')
            try:
                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))
                i += 1
            except:
                continue
    print('total %d to rename & converted %d jpgs' % (total_num_file, i))


def filtration_data(file_path, target_path, other_file_path, haarcascade_eye, haarcascade_mouth):
    """

    :param file_path: 人脸数据路径
    :param target_path: 清洗后人脸数据路径
    :param other_file_path: 被清洗后垃圾数据路径
    :param haarcascade_eye: haarcascade_eye.xml文件路径
    :param haarcascade_mouth: haarcascade_mcs_mouth,xml文件路径
    :return: None
    """
    files = os.listdir(file_path)
    # 目标文件夹判断若不存在则新建一个
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    if not os.path.exists(other_file_path):
        os.mkdir(other_file_path)

    trueDataNum = 0
    falseDataNum = 0
    print("清洗前图片 %d 张" % len(files))

    for f in files:

        # 处理图片失效问题，如果图片失效则直接跳过
        if imghdr.what(file_path + '/' + f) == None:
            continue

        # 对于imread无法读取中文路径的解决办法
        img = cv2.imdecode(numpy.fromfile(file_path + '/' + f, dtype=numpy.uint8), -1)

        # 定义分类器（人眼识别）
        classifier_eye = cv2.CascadeClassifier(haarcascade_eye)
        # 定义分类器（嘴巴识别）
        # classifier_mouth = cv2.CascadeClassifier(haarcascade_mouth)

        # 检测器识别眼睛
        # 检测器：detectMultiScale参数（图像，每次缩小图像的比例，匹配成功所需要的周围矩形框的数目，检测的类型，匹配物体的大小范围）
        faceRects_eye = classifier_eye.detectMultiScale(img, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, (20, 20))
        # 检测器识别嘴
        # 这个嘴的检测器检测效果相对来说不好不建议使用先
        # faceRects_mouth = classifier_mouth.detectMultiScale(img, 1.1, 1, cv2.CASCADE_SCALE_IMAGE, (5, 20))

        # 检测到眼睛后将该数据copy/move到filtrationData中
        # if len(faceRects_eye) > 0 and len(faceRects_mouth) == 0:
        # 在测试过程中发现只检测到奇数人眼相对来说出现误检可能性更高所以以检测到偶数人眼为准
        if len(faceRects_eye) % 2 != 1 and len(faceRects_eye) > 0:
            # shutil.copy(file_path + '/' + f, target_path)
            shutil.move(file_path + '/' + f, target_path)
            trueDataNum += 1
        else:
            # shutil.copy(file_path + '/' + f, other_file_path)
            shutil.move(file_path + '/' + f, other_file_path)
            falseDataNum += 1

    print("清洗后获得人脸数据 %d 张" % trueDataNum)
    print("过滤的图片数据 %d 张" % falseDataNum)
    print("失效图片 %d 张" % (len(files) - trueDataNum - falseDataNum))

if __name__ == '__main__':
    file_path = "./oriFace"
    target_path = "./filtrationFace"
    other_file_path = "./otherData"
    haarcascade_eye = "./haarcascade_eye.xml"
    haarcascade_mouth = "./haarcascade_mcs_mouth.xml"
    filtration_data(file_path, target_path, other_file_path, haarcascade_eye, haarcascade_mouth)
    # rename_data(target_path)
    # rename_data(other_file_path)