#!/usr/bin/env python
#-*- coding:utf-8 -*-
import ctypes
import random
import re

import pymssql
import win32con
import win32gui
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QMainWindow
from PyQt5.QtCore import Qt, QPoint, QPropertyAnimation, QTimer
from PyQt5.QtGui import QFont, QCursor, QPixmap, QMovie, QPainter, QPen, QColor, QBrush, QPainterPath, QKeyEvent, QIcon, \
    QImage
import numpy as np
import cv2
import mediapipe as mp
from PyQt5.QtWidgets import QApplication
import sys
import time
import  matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw
import math
import skimage.exposure

class RECT(ctypes.Structure):
    _fields_ = [('left', ctypes.c_int),
                ('top', ctypes.c_int),
                ('right', ctypes.c_int),
                ('bottom', ctypes.c_int)]

class facethinV:
    radius = 1  # 参数可调
    Strength = 1  # 参数可调

    def process_frame(self,process):
        mp_face_mesh = mp.solutions.face_mesh
        # help(mp_face_mesh.FaceMesh)

        model = mp_face_mesh.FaceMesh(
            static_image_mode=False,  # TRUE:静态图片/False:摄像头实时读取
            refine_landmarks=True,  # 使用Attention Mesh模型
            max_num_faces=40,
            min_detection_confidence=0.5,  # 置信度阈值，越接近1越准
            min_tracking_confidence=0.5,  # 追踪阈值
        )
        # 导入可视化函数和可视化样式
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        draw_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=[223, 155, 6])
        start_time = time.time()
        # 获取图像宽高
        h, w = process.shape[0], process.shape[1]
        process_copy = process
        # BGR转RGB
        img_RGB = cv2.cvtColor(process, cv2.COLOR_BGR2RGB)
        results = model.process(img_RGB)
        if results.multi_face_landmarks:
            left_face_up = []
            left_face_down = []
            right_face_up = []
            right_face_down = []
            endP = []
            for face_landmarks in results.multi_face_landmarks:
                for id, lm in enumerate(face_landmarks.landmark):
                    # print(f"(id,lm) = {id},{lm}")
                    h, w, c = process.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    if id == 234:
                        left_face_up.append([x, y])
                    elif id == 58:
                        left_face_down.append([x, y])
                    elif id == 288:  # 454
                        right_face_up.append([x, y])
                    elif id == 454:  # 379
                        right_face_down.append([x, y])
                    elif id == 197:
                        endP.append([x, y])
                # r_left = math.sqrt(abs((left_face_up[0][0]-left_face_down[0][0])*(left_face_up[0][0]-left_face_down[0][0])+
                # (left_face_up[0][1]-left_face_down[0][1])*(left_face_up[0][1]-left_face_down[0][1])))
                # r_right = math.sqrt(abs((right_face_up[0][0]-right_face_down[0][0])*(right_face_up[0][0]-right_face_down[0][0])+
                # (right_face_up[0][1]-right_face_down[0][1])*(right_face_up[0][1]-right_face_down[0][1])))

                thin_image = self.localTranslationWarp(process_copy, left_face_up[0][0], left_face_up[0][1],endP[0][0], endP[0][1], self.radius,self.Strength)
                thin_image = self.localTranslationWarp(thin_image, right_face_up[0][0], right_face_up[0][1],endP[0][0], endP[0][1], self.radius,self.Strength)
                # print(id,x,y)
                # print(uppoint)

        else:
            print('未检测出人脸')
        return thin_image

    def localTranslationWarp(self,srcImg, startX, startY, endX, endY, radius, Strength):
        ddradius = float(radius * radius)
        copyImg = np.zeros(srcImg.shape, np.uint8)
        copyImg = srcImg.copy()

        maskImg = np.zeros(srcImg.shape[:2], np.uint8)
        cv2.circle(maskImg, (startX, startY), math.ceil(radius), (255, 255, 255), -1)

        K0 = 100 / Strength

        # 计算公式中的|m-c|^2
        ddmc_x = (endX - startX) * (endX - startX)
        ddmc_y = (endY - startY) * (endY - startY)
        H, W, C = srcImg.shape
        # 重新把像素按行顺序分布在1~-1间,处理数据方便计算
        mapX = np.vstack([np.arange(W).astype(np.float32).reshape(1, -1)] * H)
        # 重新把像素按列顺序分布在1~-1间,处理数据方便计算
        mapY = np.hstack([np.arange(H).astype(np.float32).reshape(-1, 1)] * W)

        distance_x = (mapX - startX) * (mapX - startX)
        distance_y = (mapY - startY) * (mapY - startY)
        distance = distance_x + distance_y
        K1 = np.sqrt(distance)
        ratio_x = (ddradius - distance_x) / (ddradius - distance_x + K0 * ddmc_x)
        ratio_y = (ddradius - distance_y) / (ddradius - distance_y + K0 * ddmc_y)
        ratio_x = ratio_x * ratio_x
        ratio_y = ratio_y * ratio_y
        # 映射到原来的位置
        UX = mapX - ratio_x * (endX - startX) * (1 - K1 / radius)
        UY = mapY - ratio_y * (endY - startY) * (1 - K1 / radius)
        # 将mapX的中为0的值复制给UX
        np.copyto(UX, mapX, where=maskImg == 0)
        np.copyto(UY, mapY, where=maskImg == 0)
        # 设定值类型
        UX = UX.astype(np.float32)
        UY = UY.astype(np.float32)
        # 重映射,interpolation=cv2.INTER_LINEAR表示插值方式是双线性插值,在.remap中不支持INTER_AREA插值方式
        copyImg = cv2.remap(srcImg, UX, UY, interpolation=cv2.INTER_LINEAR)

        return copyImg

class bigeyeV:
    Radius, Strength = 1,1  # 4个参数可以独立可输入,也可设置为2个分别为Radius和Strength

    def big_eye_adjust_fast(self,src, PointX, PointY, Radius, Strength):
        processed_image = np.zeros(src.shape, np.uint8)
        processed_image = src.copy()
        height = src.shape[0]
        width = src.shape[1]
        PowRadius = Radius * Radius

        maskImg = np.zeros(src.shape[:2], np.uint8)
        cv2.circle(maskImg, (PointX, PointY), math.ceil(Radius), (255, 255, 255), -1)

        mapX = np.vstack([np.arange(width).astype(np.float32).reshape(1, -1)] * height)
        mapY = np.hstack([np.arange(height).astype(np.float32).reshape(-1, 1)] * width)

        OffsetX = mapX - PointX
        OffsetY = mapY - PointY
        XY = OffsetX * OffsetX + OffsetY * OffsetY

        ScaleFactor = 1 - XY / PowRadius
        ScaleFactor = 1 - Strength / 100 * ScaleFactor
        UX = OffsetX * ScaleFactor + PointX
        UY = OffsetY * ScaleFactor + PointY
        UX[UX < 0] = 0
        UX[UX >= width] = width - 1
        UY[UY < 0] = 0
        UY[UY >= height] = height - 1

        np.copyto(UX, mapX, where=maskImg == 0)
        np.copyto(UY, mapY, where=maskImg == 0)

        UX = UX.astype(np.float32)
        UY = UY.astype(np.float32)

        processed_image = cv2.remap(src, UX, UY, interpolation=cv2.INTER_LINEAR)

        return processed_image

    def process_frame(self,process):
        # 导入三维人脸关键点检测模型
        mp_face_mesh = mp.solutions.face_mesh
        # help(mp_face_mesh.FaceMesh)

        model = mp_face_mesh.FaceMesh(
            static_image_mode=True,  # TRUE:静态图片/False:摄像头实时读取
            refine_landmarks=True,  # 使用Attention Mesh模型
            max_num_faces=40,
            min_detection_confidence=0.5,  # 置信度阈值，越接近1越准
            min_tracking_confidence=0.5,  # 追踪阈值
        )
        # 导入可视化函数和可视化样式
        mp_drawing = mp.solutions.drawing_utils
        # mp_drawing_styles=mp.solutions.drawing_styles
        draw_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=[223, 155, 6])
        start_time = time.time()
        # 获取图像宽高
        h, w = process.shape[0], process.shape[1]

        # BGR转RGB
        img_RGB = cv2.cvtColor(process, cv2.COLOR_BGR2RGB)
        results = model.process(img_RGB)
        if results.multi_face_landmarks:
            '''
            face_num = len(results.multi_face_landmarks) # 检测出的人脸个数
            color_list = [(255,255,255),(223,155,6),(94,218,121),(180,187,28),(77,169,10),(1,240,255)]
            for i in range(face_num): # 遍历每一个人脸框
                top_X = int(results.multi_face_landmarks[i].location_data.relative_bounding_box.xmin * w)
                top_Y = int(results.multi_face_landmarks[i].location_data.relative_bounding_box.ymin * h)
                bbox_width = int(results.multi_face_landmarks[i].location_data.relative_bounding_box.width * w)
                bbox_height = int(results.multi_face_landmarks[i].location_data.relative_bounding_box.height * h)
                # 绘制人脸矩形框，左上角坐标，右下角坐标，颜色，线宽
                img = cv2.rectangle(img, (top_X, top_Y),(top_X+bbox_width, top_Y+bbox_height),color_list[i%face_num],10)
            '''
            xc = []
            yc = []
            for face_landmarks in results.multi_face_landmarks:
                '''
                mp_drawing.draw_landmarks(img, 
                                face_landmarks ,
                                #bbox_drawing_spec=bbox_style,
                                #
                                # landmark_drawing_spec=draw_spec
                                )
                '''
                for id, lm in enumerate(face_landmarks.landmark):
                    # print(f"(id,lm) = {id},{lm}")
                    h, w, c = process.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    xc.append(x)
                    yc.append(y)
                    # print(id, x, y)
                # id=160:是左眼的顶端位置
                # id=385:是右眼的top位置
                # id=472:是左眼的bottom
                # id=477:是右眼的bottom
                # if id == 477:
                # img = cv2.putText(img,str(id),(x,y),cv.FONT_HERSHEY_SIMPLEX,0.3,(218,112,214),1,1)
                # print(int(xc[160]), int(yc[472]))
                # print(int(xc[385]), int(yc[477]))
                PointX_left, PointY_left = int(xc[160]), int(yc[472])  # 19.78
                PointX_right, PointY_right = int(xc[385]), int(yc[477])
                processed_image = self.big_eye_adjust_fast(process, PointX_left, PointY_left, self.Radius,
                                                              self.Strength)
                processed_image = self.big_eye_adjust_fast(processed_image, PointX_right, PointY_right, self.Radius,
                                                              self.Strength)
        else:
            print('未检测出人脸')

        return processed_image

class lipsChangeV:
    desired_color = (0,0,0)  # 唇色设定,可调节

    def GetFace(self,image):

        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection()

        # image = cv2.imread('./face_recognition/1.ag.14.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)
        face = results.detections[0]  # 检测第一张脸
        # 人脸画框
        x1 = int(face.location_data.relative_bounding_box.xmin * image.shape[1])
        y1 = int(face.location_data.relative_bounding_box.ymin * image.shape[0])
        x2 = int(x1 + face.location_data.relative_bounding_box.width * image.shape[1])
        y2 = int(y1 + face.location_data.relative_bounding_box.height * image.shape[0])
        # 计算最大面积-->最大框
        size = max(x2 - x1, y2 - y1)
        # 中心点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        # 计算边框坐标
        x1_square = center_x - size // 2
        y1_square = center_y - size // 2
        x2_square = x1_square + size
        y2_square = y1_square + size
        # 裁剪人脸
        square_face_region = image[y1_square:y2_square, x1_square:x2_square]
        resized_image = cv2.resize(square_face_region, (480, 480))
        resized_image_bgr = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
        # Save the image
        # cv2.imwrite('resized_image.jpg', resized_image_bgr)
        print("生成成功")
        return resized_image_bgr

    def GetMask(self,resized_image_bgr):
        # 掩码生成
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, min_detection_confidence=0.5)
        image = resized_image_bgr
        # LIPS = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))
        # upper = [409,405,375,321,314,267,269,270,291,146,181,185,91,84,61,37, 39, 40,0,17]
        # lower = [402,415,312,311,310,308,324,318,317,178,191,80, 81, 82,87, 88,95,78,13, 14]
        upper_new = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]
        lower_new = [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82]
        results = face_mesh.process(image)
        mask_upper = np.zeros(image.shape[:2], dtype=np.uint8)
        # 填充上嘴唇
        for face_landmarks in results.multi_face_landmarks:
            points_upper = []
            for i in upper_new:
                landmark = face_landmarks.landmark[i]
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                points_upper.append((x, y))
            cv2.fillConvexPoly(mask_upper, np.int32(points_upper), 255)
        # 填充下嘴唇
        mask_lower = np.zeros(image.shape[:2], dtype=np.uint8)
        for face_landmarks in results.multi_face_landmarks:
            points_lower = []
            for i in lower_new:
                landmark = face_landmarks.landmark[i]
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                points_lower.append((x, y))
            cv2.fillPoly(mask_lower, np.int32([points_lower]), 255)
        #  上下嘴唇掩膜分开处理-->为了得到更平滑的边缘
        mask_diff = cv2.subtract(mask_upper, mask_lower)
        # 平滑掩膜
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_diff = cv2.morphologyEx(mask_diff, cv2.MORPH_OPEN, kernel)
        mask_diff = cv2.morphologyEx(mask_diff, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('result',mask_diff)
        # cv2.imwrite('mask_diff.jpg', mask_diff)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return mask_diff

    def LipsColar(self,process, mask_diff):
        # desired_color = (28,28,28)        #可根据参数输入自己想要的颜色
        # print(self.desired_color)

        # swatch = np.full((200,200,3), self.desired_color, dtype=np.uint8)

        img = process
        # img = cv2.imread("resized_image.jpg")

        mask = mask_diff
        # mask = cv2.imread("mask_diff.jpg", cv2.IMREAD_GRAYSCALE)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        # 得到平均值
        ave_color = cv2.mean(img, mask=mask)[:3]
        # print(ave_color)
        ave_color_img = np.full((1, 1, 3), ave_color, dtype=np.float32)
        # print(ave_color_img)
        desired_color_img = np.full((1, 1, 3), self.desired_color, dtype=np.float32)
        # print(desired_color_img)
        # 将bgr颜色转为hsv
        desired_hsv = cv2.cvtColor(desired_color_img, cv2.COLOR_BGR2HSV)
        ave_hsv = cv2.cvtColor(ave_color_img, cv2.COLOR_BGR2HSV)
        diff_hsv = desired_hsv - ave_hsv
        diff_h, diff_s, diff_v = cv2.split(diff_hsv)
        # print(diff_hsv)
        # 分别给3个通道赋值
        hnew = np.mod(h + diff_h / 2, 180).astype(np.uint8)
        snew = (s + diff_s).clip(0, 255).astype(np.uint8)
        vnew = (v + diff_v).clip(0, 255).astype(np.uint8)
        # 各个通道融合
        hsv_new = cv2.merge([hnew, snew, vnew])
        # 将hsv图像转化成BGR
        new_img = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
        # 归一化数值（0~1）,生成3通道
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=5, sigmaY=5, borderType=cv2.BORDER_DEFAULT)
        mask = skimage.exposure.rescale_intensity(mask, in_range=(128, 255), out_range=(0, 1)).astype(np.float32)
        mask = cv2.merge([mask, mask, mask])
        # 将掩膜和过程图融合
        result = (img * (1 - mask) + new_img * mask)
        result = result.clip(0, 255).astype(np.uint8)
        # save result
        # cv2.imwrite('lady2_swatch.png', swatch)
        # cv2.imwrite('lady2_recolor.jpg', result)
        # cv2.imshow('swatch', swatch)
        # cv2.imshow('mask', mask)
        # cv2.imshow('new_img', new_img)
        # cv2.imshow('result', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return result

    def process_frame(self,process):
        # resized_image_bgr = GetFace(process)
        mask_diff = self.GetMask(process)
        Rimg = self.LipsColar(process, mask_diff)
        return Rimg

class Beauty(QWidget):
    ui=None
    numbers=[]
    def __init__(self,ui):
        self.ui=ui
        super(Beauty, self).__init__()
        f = open('C:/Users/林逸/Desktop/综合课设三/ini/%s.ini' % self.ui.id, 'r')
        with f:
            text = f.read()
            pattern = r'\d+'
            number = re.findall(pattern, text)
            self.numbers = [int(num) for num in number]
            f.close()
        self.setObjectName("塑颜美颜系统")
        self.resize(ui.size().width(), 300)
        self.setWindowTitle('塑颜美颜系统')
        self.setWindowIcon(QIcon(QPixmap('C:/Users/林逸/Desktop/综合课设三/resource/卡通.png')))
        self.move(ui.pos().x(), ui.pos().y()+ui.size().height())
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setStyleSheet('color: white;')

        self.label=QLabel(self)
        self.label.setObjectName('label')
        self.label.resize(self.size().width(),self.size().height())
        self.label.move(0,0)
        movie=QMovie('./resource/3.gif')
        movie.setScaledSize(QtCore.QSize(self.size().width(),self.size().height()))
        self.label.setMovie(movie)
        movie.start()
        self.label.lower()


        font = QFont()
        font.setFamily("华康方圆体W7")
        font.setPointSize(8)
        self.checkBox = QtWidgets.QCheckBox(self)
        self.checkBox.setGeometry(QtCore.QRect(280, 350, 91, 19))
        self.checkBox.setObjectName("checkBox")
        self.checkBox.setText('''美白
（美白系数）''')
        self.checkBox.setChecked(False)
        self.checkBox.setStyleSheet("QCheckBox::indicator {color: rgb(204, 204, 204);}")
        self.checkBox.move(20,0)
        self.checkBox.resize(148, 40)
        self.checkBox.setFont(font)
        self.checkBox.stateChanged.connect(self.white)

        self.checkBox_2 = QtWidgets.QCheckBox(self)
        self.checkBox_2.setGeometry(QtCore.QRect(280, 350, 91, 19))
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_2.setText('''瘦脸
（radius/Strength）''')
        self.checkBox_2.setChecked(False)
        self.checkBox_2.setStyleSheet("QCheckBox::indicator {color: rgb(204, 204, 204);}")
        self.checkBox_2.move(20, 60)
        self.checkBox_2.resize(148, 40)
        self.checkBox_2.setFont(font)
        self.checkBox_2.stateChanged.connect(self.face)

        self.checkBox_3 = QtWidgets.QCheckBox(self)
        self.checkBox_3.setGeometry(QtCore.QRect(280, 350, 91, 19))
        self.checkBox_3.setObjectName("checkBox_3")
        self.checkBox_3.setText('''大眼
（radius/Strength）''')
        self.checkBox_3.setChecked(False)
        self.checkBox_3.setStyleSheet("QCheckBox::indicator {color: rgb(204, 204, 204);}")
        self.checkBox_3.move(20, 120)
        self.checkBox_3.resize(148, 40)
        self.checkBox_3.setFont(font)
        self.checkBox_3.stateChanged.connect(self.eye)

        self.checkBox_4 = QtWidgets.QCheckBox(self)
        self.checkBox_4.setGeometry(QtCore.QRect(280, 350, 91, 19))
        self.checkBox_4.setObjectName("checkBox_4")
        self.checkBox_4.setText('''美妆
（口红色号B,G,R）''')
        self.checkBox_4.setChecked(False)
        self.checkBox_4.setStyleSheet("QCheckBox::indicator {color: rgb(204, 204, 204);}")
        self.checkBox_4.move(20, 205)
        self.checkBox_4.resize(148, 40)
        self.checkBox_4.setFont(font)
        self.checkBox_4.stateChanged.connect(self.lip)


        self.horizontalSlider_0 = QtWidgets.QSlider(self)
        self.horizontalSlider_0.setGeometry(QtCore.QRect(450, 350, 160, 22))
        self.horizontalSlider_0.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_0.setObjectName("horizontalSlider_0")
        self.horizontalSlider_0.resize(350,20)
        self.horizontalSlider_0.move(160,10)
        self.horizontalSlider_0.setStyleSheet('QSlider::groove:horizontal {border: 0px solid #bbb;} QSlider::sub-page:horizontal {background: rgb(84,104,197);border-radius: 2px;margin-top:8px;margin-bottom:8px;}QSlider::add-page:horizontal {background: rgb(255,255, 255);border: 0px solid #777;border-radius: 2px;margin-top:9px;margin-bottom:9px;}QSlider::handle:horizontal {background: rgb(193,204,208);width: 5px;border: 1px solid rgb(193,204,208);border-radius: 2px; margin-top:6px;margin-bottom:6px;}QSlider::handle:horizontal:hover {background: rgb(193,204,208);width: 10px;border: 1px solid rgb(193,204,208);border-radius: 5px; margin-top:4px;margin-bottom:4px;}')
        self.horizontalSlider_0.setRange(0,20)
        self.horizontalSlider_0.setValue(self.numbers[0]*10)
        self.horizontalSlider_0.valueChanged.connect(self.change)

        self.horizontalSlider_1 = QtWidgets.QSlider(self)
        self.horizontalSlider_1.setGeometry(QtCore.QRect(450, 350, 160, 22))
        self.horizontalSlider_1.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_1.setObjectName("horizontalSlider_1")
        self.horizontalSlider_1.resize(350,20)
        self.horizontalSlider_1.move(160,self.horizontalSlider_0.pos().y()+45)
        self.horizontalSlider_1.setStyleSheet('QSlider::groove:horizontal {border: 0px solid #bbb;} QSlider::sub-page:horizontal {background: rgb(84,104,197);border-radius: 2px;margin-top:8px;margin-bottom:8px;}QSlider::add-page:horizontal {background: rgb(255,255, 255);border: 0px solid #777;border-radius: 2px;margin-top:9px;margin-bottom:9px;}QSlider::handle:horizontal {background: rgb(193,204,208);width: 5px;border: 1px solid rgb(193,204,208);border-radius: 2px; margin-top:6px;margin-bottom:6px;}QSlider::handle:horizontal:hover {background: rgb(193,204,208);width: 10px;border: 1px solid rgb(193,204,208);border-radius: 5px; margin-top:4px;margin-bottom:4px;}')
        self.horizontalSlider_1.setRange(1,40)
        self.horizontalSlider_1.setValue(self.numbers[1])
        self.horizontalSlider_1.valueChanged.connect(self.change)

        self.horizontalSlider_2 = QtWidgets.QSlider(self)
        self.horizontalSlider_2.setGeometry(QtCore.QRect(450, 350, 160, 22))
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.horizontalSlider_2.resize(350,20)
        self.horizontalSlider_2.move(160,self.horizontalSlider_1.pos().y()+30)
        self.horizontalSlider_2.setStyleSheet('QSlider::groove:horizontal {border: 0px solid #bbb;} QSlider::sub-page:horizontal {background: rgb(84,104,197);border-radius: 2px;margin-top:8px;margin-bottom:8px;}QSlider::add-page:horizontal {background: rgb(255,255, 255);border: 0px solid #777;border-radius: 2px;margin-top:9px;margin-bottom:9px;}QSlider::handle:horizontal {background: rgb(193,204,208);width: 5px;border: 1px solid rgb(193,204,208);border-radius: 2px; margin-top:6px;margin-bottom:6px;}QSlider::handle:horizontal:hover {background: rgb(193,204,208);width: 10px;border: 1px solid rgb(193,204,208);border-radius: 5px; margin-top:4px;margin-bottom:4px;}')
        self.horizontalSlider_2.setRange(1,90)
        self.horizontalSlider_2.setValue(self.numbers[2])
        self.horizontalSlider_2.valueChanged.connect(self.change)

        self.horizontalSlider_3 = QtWidgets.QSlider(self)
        self.horizontalSlider_3.setGeometry(QtCore.QRect(450, 350, 160, 22))
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        self.horizontalSlider_3.resize(350,20)
        self.horizontalSlider_3.move(160,self.horizontalSlider_2.pos().y()+30)
        self.horizontalSlider_3.setStyleSheet('QSlider::groove:horizontal {border: 0px solid #bbb;} QSlider::sub-page:horizontal {background: rgb(84,104,197);border-radius: 2px;margin-top:8px;margin-bottom:8px;}QSlider::add-page:horizontal {background: rgb(255,255, 255);border: 0px solid #777;border-radius: 2px;margin-top:9px;margin-bottom:9px;}QSlider::handle:horizontal {background: rgb(193,204,208);width: 5px;border: 1px solid rgb(193,204,208);border-radius: 2px; margin-top:6px;margin-bottom:6px;}QSlider::handle:horizontal:hover {background: rgb(193,204,208);width: 10px;border: 1px solid rgb(193,204,208);border-radius: 5px; margin-top:4px;margin-bottom:4px;}')
        self.horizontalSlider_3.setRange(1,100)
        self.horizontalSlider_3.setValue(self.numbers[3])
        self.horizontalSlider_3.valueChanged.connect(self.change)

        self.horizontalSlider_4 = QtWidgets.QSlider(self)
        self.horizontalSlider_4.setGeometry(QtCore.QRect(450, 350, 160, 22))
        self.horizontalSlider_4.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_4.setObjectName("horizontalSlider_4")
        self.horizontalSlider_4.resize(350,20)
        self.horizontalSlider_4.move(160,self.horizontalSlider_3.pos().y()+30)
        self.horizontalSlider_4.setStyleSheet('QSlider::groove:horizontal {border: 0px solid #bbb;} QSlider::sub-page:horizontal {background: rgb(84,104,197);border-radius: 2px;margin-top:8px;margin-bottom:8px;}QSlider::add-page:horizontal {background: rgb(255,255, 255);border: 0px solid #777;border-radius: 2px;margin-top:9px;margin-bottom:9px;}QSlider::handle:horizontal {background: rgb(193,204,208);width: 5px;border: 1px solid rgb(193,204,208);border-radius: 2px; margin-top:6px;margin-bottom:6px;}QSlider::handle:horizontal:hover {background: rgb(193,204,208);width: 10px;border: 1px solid rgb(193,204,208);border-radius: 5px; margin-top:4px;margin-bottom:4px;}')
        self.horizontalSlider_4.setRange(1,100)
        self.horizontalSlider_4.setValue(self.numbers[4])
        self.horizontalSlider_4.valueChanged.connect(self.change)

        self.horizontalSlider_5 = QtWidgets.QSlider(self)
        self.horizontalSlider_5.setGeometry(QtCore.QRect(450, 350, 160, 22))
        self.horizontalSlider_5.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_5.setObjectName("horizontalSlider_5")
        self.horizontalSlider_5.resize(350,20)
        self.horizontalSlider_5.move(160,self.horizontalSlider_4.pos().y()+45)
        self.horizontalSlider_5.setStyleSheet('QSlider::groove:horizontal {border: 0px solid #bbb;} QSlider::sub-page:horizontal {background: rgb(84,104,197);border-radius: 2px;margin-top:8px;margin-bottom:8px;}QSlider::add-page:horizontal {background: rgb(255,255, 255);border: 0px solid #777;border-radius: 2px;margin-top:9px;margin-bottom:9px;}QSlider::handle:horizontal {background: rgb(193,204,208);width: 5px;border: 1px solid rgb(193,204,208);border-radius: 2px; margin-top:6px;margin-bottom:6px;}QSlider::handle:horizontal:hover {background: rgb(193,204,208);width: 10px;border: 1px solid rgb(193,204,208);border-radius: 5px; margin-top:4px;margin-bottom:4px;}')
        self.horizontalSlider_5.setRange(0,255)
        self.horizontalSlider_5.setValue(self.numbers[5])
        self.horizontalSlider_5.valueChanged.connect(self.change)

        self.horizontalSlider_6 = QtWidgets.QSlider(self)
        self.horizontalSlider_6.setGeometry(QtCore.QRect(450, 350, 160, 22))
        self.horizontalSlider_6.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_6.setObjectName("horizontalSlider_6")
        self.horizontalSlider_6.resize(350, 20)
        self.horizontalSlider_6.move(160, self.horizontalSlider_5.pos().y() + 30)
        self.horizontalSlider_6.setStyleSheet(
            'QSlider::groove:horizontal {border: 0px solid #bbb;} QSlider::sub-page:horizontal {background: rgb(84,104,197);border-radius: 2px;margin-top:8px;margin-bottom:8px;}QSlider::add-page:horizontal {background: rgb(255,255, 255);border: 0px solid #777;border-radius: 2px;margin-top:9px;margin-bottom:9px;}QSlider::handle:horizontal {background: rgb(193,204,208);width: 5px;border: 1px solid rgb(193,204,208);border-radius: 2px; margin-top:6px;margin-bottom:6px;}QSlider::handle:horizontal:hover {background: rgb(193,204,208);width: 10px;border: 1px solid rgb(193,204,208);border-radius: 5px; margin-top:4px;margin-bottom:4px;}')
        self.horizontalSlider_6.setRange(0, 255)
        self.horizontalSlider_6.setValue(self.numbers[6])
        self.horizontalSlider_6.valueChanged.connect(self.change)

        self.horizontalSlider_7 = QtWidgets.QSlider(self)
        self.horizontalSlider_7.setGeometry(QtCore.QRect(450, 350, 160, 22))
        self.horizontalSlider_7.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_7.setObjectName("horizontalSlider_7")
        self.horizontalSlider_7.resize(350, 20)
        self.horizontalSlider_7.move(160, self.horizontalSlider_6.pos().y() + 30)
        self.horizontalSlider_7.setStyleSheet(
            'QSlider::groove:horizontal {border: 0px solid #bbb;} QSlider::sub-page:horizontal {background: rgb(84,104,197);border-radius: 2px;margin-top:8px;margin-bottom:8px;}QSlider::add-page:horizontal {background: rgb(255,255, 255);border: 0px solid #777;border-radius: 2px;margin-top:9px;margin-bottom:9px;}QSlider::handle:horizontal {background: rgb(193,204,208);width: 5px;border: 1px solid rgb(193,204,208);border-radius: 2px; margin-top:6px;margin-bottom:6px;}QSlider::handle:horizontal:hover {background: rgb(193,204,208);width: 10px;border: 1px solid rgb(193,204,208);border-radius: 5px; margin-top:4px;margin-bottom:4px;}')
        self.horizontalSlider_7.setRange(0, 255)
        self.horizontalSlider_7.setValue(self.numbers[7])
        self.horizontalSlider_7.valueChanged.connect(self.change)

        self.change()

    def white(self):
        state = self.checkBox.checkState()
        if state == Qt.Checked:
            self.ui.white=1
        else:
            self.ui.white = 0

    def face(self):
        state = self.checkBox_2.checkState()
        if state == Qt.Checked:
            self.ui.face = 1
        else:
            self.ui.face = 0

    def eye(self):
        state = self.checkBox_3.checkState()
        if state == Qt.Checked:
            self.ui.eye = 1
        else:
            self.ui.eye = 0

    def lip(self):
        state = self.checkBox_4.checkState()
        if state == Qt.Checked:
            self.ui.lip = 1
        else:
            self.ui.lip = 0

    def change(self):
        self.ui.gamma=self.horizontalSlider_0.value()/10
        facethinV.radius=self.horizontalSlider_1.value()
        facethinV.Strength=self.horizontalSlider_2.value()
        bigeyeV.Radius=self.horizontalSlider_3.value()
        bigeyeV.Strength=self.horizontalSlider_4.value()
        lipsChangeV.desired_color=(self.horizontalSlider_5.value(),self.horizontalSlider_6.value(),self.horizontalSlider_7.value())
        f = open('C:/Users/林逸/Desktop/综合课设三/ini/%s.ini' % self.ui.id, 'w')
        with f:
            f.write(
'''white: %s
shou_radius: %s
shou_Strength: %s
eye_radius: %s
eye_Strength: %s
lip_B: %s
lip_G: %s
lip_R: %s'''%(self.horizontalSlider_0.value()/10,self.horizontalSlider_1.value(),self.horizontalSlider_2.value(),self.horizontalSlider_3.value(),self.horizontalSlider_4.value(),self.horizontalSlider_5.value(),self.horizontalSlider_6.value(),self.horizontalSlider_7.value(),))
            f.close()
class Title(QLabel):
    def __init__(self, *args):
        super(Title, self).__init__(*args)
        self.setStyleSheet("QLabel{color:rgb(0,255,0,0);}")
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setFixedHeight(30)

class TitleButton(QPushButton):
    def __init__(self, *args):
        super(TitleButton, self).__init__(*args)
        self.setFont(QFont("Webdings"))
        self.setFixedWidth(40)

class UI(QWidget):
    ui=None
    gamma=0
    white=0
    face=0
    eye=0
    lip=0
    id=None
    def __init__(self,id):
        super(UI, self).__init__()
        self.id=id
        self.resize(537, 580)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self._padding = 5
        self._TitleLabel = Title(self)
        self._TitleLabel.setMouseTracking(True)
        self._TitleLabel.setIndent(10)
        self._TitleLabel.move(0, 0)
        self.label = QtWidgets.QLabel(self)
        self.label.setObjectName("label")
        self.label.resize(537, 42)
        self.label.move(0, 0)
        self.label.setStyleSheet("QLabel { background-color: rgb(58,119,253,50); }")
        # movie=QMovie('C:/Users/林逸/Desktop/综合课设三/resource/3.gif')
        # movie.setScaledSize(QtCore.QSize(self.label.size().width(),self.label.size().height()))
        # self.label.setMovie(movie)
        # movie.start()
        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setObjectName("label_2")
        self.label_2.resize(30, 30)
        self.label_2.move(5, 7)
        pix = QPixmap('C:/Users/林逸/Desktop/综合课设三/resource/卡通人像.png')
        self.label_2.setPixmap(pix)
        self.label_2.setScaledContents(True)
        self.label_3 = QtWidgets.QLabel(self)
        self.label_3.setObjectName("label_3")
        self.label_3.resize(200, 30)
        self.label_3.move(35, 7)
        self.label_3.setText('塑颜相机')
        font = QFont()
        font.setFamily("华康方圆体W7")
        font.setPointSize(15)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("QLabel { color: rgb(255,255,255); }")
        self.button=QPushButton(self)
        self.button.setObjectName('button')
        self.button.resize(30,30)
        self.button.move(self.size().width()-self.button.size().width(),self.size().height()-self.button.size().height())
        self.button.setText(b'\xef\x81\xb8'.decode("utf-8"))
        self.button.setStyleSheet("QPushButton { background-color: transparent; }")
        self.button.setFont(font)
        self.label_4=QLabel(self)
        self.label_4.setObjectName("label_3")
        self.label_4.resize(self.size().width(),self.size().width()-self.label.size().height()+13)
        self.label_4.move(0, self.label.size().height())
        self.label_4.setScaledContents(True)
        self.label_4.setStyleSheet("border: none;")
        self.window = self
        self._MainLayout = QVBoxLayout()
        self._MainLayout.setSpacing(0)
        self._MainLayout.addWidget(QLabel(), Qt.AlignLeft)
        self._MainLayout.addStretch()
        self.setLayout(self._MainLayout)

        self._CloseButton = TitleButton(b'\xef\x81\xb2'.decode("utf-8"), self)
        self._CloseButton.setObjectName("CloseButton")
        self._CloseButton.setToolTip("关闭窗口")
        self._CloseButton.setMouseTracking(True)
        self._CloseButton.setFixedHeight(self._TitleLabel.height())
        self._CloseButton.clicked.connect(self.exit)
        self._MinimumButton = TitleButton(b'\xef\x80\xb0'.decode("utf-8"), self)
        self._MinimumButton.setObjectName("MinMaxButton")
        self._MinimumButton.setToolTip("最小化")
        self._MinimumButton.setMouseTracking(True)
        self._MinimumButton.setFixedHeight(self._TitleLabel.height())
        self._MinimumButton.clicked.connect(self.showMinimized)
        self.setMouseTracking(True)
        self._move_drag = False
        self.cap = cv2.VideoCapture(0)  # 打开摄像头
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh=self.mp_face_mesh.FaceMesh(max_num_faces=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)
        self.c=facethinV()
        self.y = bigeyeV()
        self.f = lipsChangeV()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.system)
        self.timer.start(0)  # 以30ms的间隔更新图像


    def exit(self):
        try:
            self.ui.close()
        except:
            pass
        self.window.close()

    def open(self):
        self.button.setText(b'\xef\x81\xb7'.decode("utf-8"))
        self.ui=Beauty(self)
        self.ui.show()

    def contract(self):
        try:
            self.button.setText(b'\xef\x81\xb8'.decode("utf-8"))
            self.ui.close()
            self.ui=None
        except:
            pass

    def system(self):
        success, image = self.cap.read()
        if self.face==1:
            # 记录该帧开始处理的时间
            image = self.c.process_frame(image)
        if self.eye==1:
            image = self.y.process_frame(image)
        if self.lip==1:
            image = self.f.process_frame(image)
        if self.white==1:
            try:
                gamma = self.gamma
                image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
                image=np.clip(image,0,255)
                image = np.power(image, gamma)
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            except:
                pass
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        pix = QPixmap.fromImage(QImage(image.data, width, height, bytesPerLine, QImage.Format_BGR888))
        self.label_4.setPixmap(pix)

    # 重写方法以实现拖拽
    def resizeEvent(self, QResizeEvent):
        self._TitleLabel.setFixedWidth(self.width())
        self._CloseButton.move(self.width() - self._CloseButton.width(), 0)
        self._MinimumButton.move(self.width() - (self._CloseButton.width() + 1) * 2 + 1, 0)

    def mousePressEvent(self, event):
        if (event.button() == Qt.LeftButton) and (event.y() < self.label.height()):
            self._move_drag = True
            self.move_DragPosition = event.globalPos() - self.pos()
            event.accept()
        elif (event.button() == Qt.LeftButton) and (event.x()>=0 and event.x()<=self.size().width()) and (event.y()>=self.size().height()-self.button.size().height() and event.y()<=self.size().height()):
            if self.ui==None:
                self.open()
            else:
                self.contract()

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self._move_drag:
            self.move(QMouseEvent.globalPos() - self.move_DragPosition)
            try:
                self.ui.move(self.pos().x(), self.pos().y()+self.size().height())
            except:
                pass
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self._move_drag = False

    def paintEvent(self, event):
        painter = QPainter()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.begin(self)
        painter.setBrush(QColor(255, 255, 255))
        # painter.drawRoundedRect(30, 60, self.width() - 60, self.height() - 80, 20, 20)
        pen = QPen(Qt.gray, 2, Qt.SolidLine)
        pen.setColor(QColor(0,0,0))
        painter.setPen(pen)
        painter.drawLine(0, self.size().height()-self.button.size().height(), self.size().width(), self.size().height()-self.button.size().height())
        painter.end()

class 注册(QWidget):
    ui=None
    num=1
    def __init__(self):
        super(注册, self).__init__()
        self.resize(537, 415)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self._padding = 5
        self._TitleLabel = Title(self)
        self._TitleLabel.setMouseTracking(True)
        self._TitleLabel.setIndent(10)
        self._TitleLabel.move(0, 0)
        self.label = QtWidgets.QLabel(self)
        self.label.setObjectName("label")
        self.label.resize(537, 45)
        self.label.move(0, 0)
        pix=QPixmap('C:/Users/林逸/Desktop/综合课设三/resource/注册.png')
        self.label.setPixmap(pix)
        self.label.setScaledContents(True)
        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setObjectName("label_2")
        self.label_2.resize(30, 30)
        self.label_2.move(5, 7)
        pix = QPixmap('C:/Users/林逸/Desktop/综合课设三/resource/安全中心.png')
        self.label_2.setPixmap(pix)
        self.label_2.setScaledContents(True)
        self.label_7 = self.label_2 = QtWidgets.QLabel(self)
        self.label_7.setObjectName("label_2")
        self.label_7.resize(200, 30)
        self.label_7.move(35, 7)
        self.label_7.setText('注册中心')
        self.label_7.setStyleSheet("QLabel { color: rgb(255,255,255); }")
        font = QFont()
        font.setFamily("华康方圆体W7")
        font.setPointSize(12)
        self.label_7.setFont(font)
        font = QFont()
        font.setFamily("幼体")
        font.setPointSize(15)
        self.label_3 = QtWidgets.QLabel(self)
        self.label_3.setGeometry(QtCore.QRect(220, 270, 72, 15))
        self.label_3.setObjectName("label_3")
        self.label_3.setText('欢迎注册塑颜')
        self.label_3.setFont(font)
        self.label_3.resize(150, 30)
        self.label_3.move(85, 80)
        font = QFont()
        font.setFamily("楷体")
        font.setPointSize(13)
        self.label_4 = QtWidgets.QLabel(self)
        self.label_4.setGeometry(QtCore.QRect(220, 310, 72, 15))
        self.label_4.setObjectName("label_4")
        self.label_4.setText('请输入账号')
        self.label_4.resize(150, 30)
        self.label_4.move(85, 140)
        self.label_4.setFont(font)
        font1 = QFont()
        font1.setFamily("楷体")
        font1.setPointSize(8)
        self.lab = QtWidgets.QLabel(self)
        self.lab.setGeometry(QtCore.QRect(220, 310, 72, 15))
        self.lab.setObjectName("lab")
        self.lab.setText("")
        self.lab.setStyleSheet("color: red")
        self.lab.resize(180, 30)
        self.lab.move(350, 140)
        self.lab.setFont(font1)
        self.lineEdit_1 = QtWidgets.QLineEdit(self)
        self.lineEdit_1.setGeometry(QtCore.QRect(230, 350, 113, 21))
        self.lineEdit_1.setStyleSheet("background-color: rgb(245,246,250);border-radius: 10px")
        self.lineEdit_1.setObjectName("lineEdit_1")
        self.lineEdit_1.resize(370, 30)
        self.lineEdit_1.move(85, 170)
        self.label_5 = QtWidgets.QLabel(self)
        self.label_5.setGeometry(QtCore.QRect(220, 310, 72, 15))
        self.label_5.setObjectName("label_4")
        self.label_5.setText('请输入密码')
        self.label_5.resize(150, 30)
        self.label_5.move(85, 200)
        self.label_5.setFont(font)
        self.lab2 = QtWidgets.QLabel(self)
        self.lab2.setGeometry(QtCore.QRect(220, 310, 72, 15))
        self.lab2.setObjectName("lab2")
        self.lab2.setText("")
        self.lab2.setStyleSheet("color: red")
        self.lab2.resize(180, 30)
        self.lab2.move(350, 200)
        self.lab2.setFont(font1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self)
        self.lineEdit_2.setGeometry(QtCore.QRect(230, 350, 113, 21))
        self.lineEdit_2.setStyleSheet("background-color: rgb(245,246,250);border-radius: 10px")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_2.resize(370, 30)
        self.lineEdit_2.move(85, 230)

        self.label_6 = QtWidgets.QLabel(self)
        self.label_6.setGeometry(QtCore.QRect(220, 310, 72, 15))
        self.label_6.setObjectName("label_4")
        self.label_6.setText('请再次输入密码')
        self.label_6.resize(180, 30)
        self.label_6.move(85, 260)
        self.label_6.setFont(font)
        self.lab3 = QtWidgets.QLabel(self)
        self.lab3.setGeometry(QtCore.QRect(220, 310, 72, 15))
        self.lab3.setObjectName("lab3")
        self.lab3.setText("")
        self.lab3.setStyleSheet("color: red")
        self.lab3.resize(180, 30)
        self.lab3.move(350, 260)
        self.lab3.setFont(font1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self)
        self.lineEdit_3.setGeometry(QtCore.QRect(230, 350, 113, 21))
        self.lineEdit_3.setStyleSheet("background-color: rgb(245,246,250);border-radius: 10px")
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_3.resize(370, 30)
        self.lineEdit_3.move(85, 290)
        self.lineEdit_2.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_3.setEchoMode(QtWidgets.QLineEdit.Password)
        font = QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        self.lineEdit_1.setFont(font)
        self.lineEdit_2.setFont(font)
        self.lineEdit_3.setFont(font)

        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setGeometry(QtCore.QRect(230, 400, 371, 28))
        self.pushButton.setStyleSheet(
            "QPushButton { background-color: rgb(67,152,253);border-radius: 10px;color: white; }")
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText('确定')
        self.pushButton.resize(370, 40)
        self.pushButton.move(85, 340)

        self._MainLayout = QVBoxLayout()
        self._MainLayout.setSpacing(0)
        self._MainLayout.addWidget(QLabel(), Qt.AlignLeft)
        self._MainLayout.addStretch()
        self.setLayout(self._MainLayout)

        self._CloseButton = TitleButton(b'\xef\x81\xb2'.decode("utf-8"), self)
        self._CloseButton.setObjectName("CloseButton")
        self._CloseButton.setToolTip("关闭窗口")
        self._CloseButton.setMouseTracking(True)
        self._CloseButton.setFixedHeight(self._TitleLabel.height())
        self._CloseButton.clicked.connect(self.close)
        self._MinimumButton = TitleButton(b'\xef\x80\xb0'.decode("utf-8"), self)
        self._MinimumButton.setObjectName("MinMaxButton")
        self._MinimumButton.setToolTip("最小化")
        self._MinimumButton.setMouseTracking(True)
        self._MinimumButton.setFixedHeight(self._TitleLabel.height())
        self._MinimumButton.clicked.connect(self.showMinimized)
        self.setMouseTracking(True)
        self._move_drag = False

    def do(self):
        server = "127.0.0.1"
        user = "sa"
        password = "hbw516626"
        database = "美颜系统"
        conn = pymssql.connect(server, user, password, database, autocommit=True)
        self.cursor = conn.cursor()
        self.cursor.execute('''
SELECT password FROM 用户
WHERE id=\'''' + self.lineEdit_1.text() + '\'')
        item=self.cursor.fetchall()
        if self.lineEdit_1.text()=='':
            self.lab.setText('错误账号')
            self.lab2.setText('')
            self.lab3.setText('')
        elif item!=[]:
            self.lab.setText('此账号已存在')
            self.lab2.setText('')
            self.lab3.setText('')
        elif self.lineEdit_2.text() == '':
            self.lab.setText('')
            self.lab2.setText('密码不可为空')
            self.lab3.setText('')
        elif self.lineEdit_2.text() != self.lineEdit_3.text():
            self.lab.setText('')
            self.lab2.setText('')
            self.lab3.setText('与上个密码不匹配')
        else:
            self.cursor.execute('''
insert into 用户
values (%s,%s)
''', (self.lineEdit_1.text(), self.lineEdit_3.text()))
            f = open('C:/Users/林逸/Desktop/综合课设三/ini/%s.ini'%self.lineEdit_1.text(), 'w')
            with f:
                f.write(
'''white: 0
shou_radius: 1
shou_Strength: 1
eye_radius: 1
eye_Strength: 1
lip_B: 0
lip_G: 0
lip_R: 0''')
                f.close()
            self.cursor.close()
            self.close()
            self.ui = 登录界面()
            self.ui.show()



    # 重写方法以实现拖拽
    def resizeEvent(self, QResizeEvent):
        self._TitleLabel.setFixedWidth(self.width())
        self._CloseButton.move(self.width() - self._CloseButton.width(), 0)
        self._MinimumButton.move(self.width() - (self._CloseButton.width() + 1) * 2 + 1, 0)

    def mousePressEvent(self, event):
        if (event.button() == Qt.LeftButton) and (event.y() < self.label.height()):
            self._move_drag = True
            self.move_DragPosition = event.globalPos() - self.pos()
            event.accept()
        elif (event.button() == Qt.LeftButton) and (event.x() >= self.lineEdit_1.pos().x() and event.x() <= self.lineEdit_1.pos().x()+self.lineEdit_1.size().width()) and (event.y() >= self.lineEdit_1.pos().y() and event.y() <= self.lineEdit_1.pos().y()+self.lineEdit_1.size().height()):
            self.lineEdit_1.setFocus()
            self.num=1
        elif (event.button() == Qt.LeftButton) and (event.x() >= self.lineEdit_2.pos().x() and event.x() <= self.lineEdit_2.pos().x()+self.lineEdit_2.size().width()) and (event.y() >= self.lineEdit_2.pos().y() and event.y() <= self.lineEdit_2.pos().y()+self.lineEdit_2.size().height()):
            self.lineEdit_2.setFocus()
            self.num=2
        elif (event.button() == Qt.LeftButton) and (event.x() >= self.lineEdit_3.pos().x() and event.x() <= self.lineEdit_3.pos().x()+self.lineEdit_3.size().width()) and (event.y() >= self.lineEdit_3.pos().y() and event.y() <= self.lineEdit_3.pos().y()+self.lineEdit_3.size().height()):
            self.lineEdit_3.setFocus()
            self.num=3
        elif (event.button() == Qt.LeftButton) and (event.x() >= 85 and event.x() <= 455) and (event.y() >= 340 and event.y() <= 380):
            self.do()

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self._move_drag:
            self.move(QMouseEvent.globalPos() - self.move_DragPosition)
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self._move_drag = False

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.do()
        elif event.key() == Qt.Key_Down:
            if self.num==3:
                self.lineEdit_1.setFocus()
                self.num=1
            elif self.num==1:
                self.lineEdit_2.setFocus()
                self.num = 2
            elif self.num==2:
                self.lineEdit_3.setFocus()
                self.num = 3
        elif event.key() == Qt.Key_Up:
            if self.num == 3:
                self.lineEdit_2.setFocus()
                self.num = 2
            elif self.num == 2:
                self.lineEdit_1.setFocus()
                self.num = 1
            elif self.num == 1:
                self.lineEdit_3.setFocus()
                self.num = 3

    def paintEvent(self, event):
        painter = QPainter()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.begin(self)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawRoundedRect(30, 60, self.width() - 60, self.height() - 80, 20, 20)
        pen = QPen(Qt.gray, 5, Qt.SolidLine)
        pen.setColor(QColor(67, 152, 253))
        painter.setPen(pen)
        # painter.drawLine(115, 255, 425, 255)
        painter.drawLine(80, 85, 80, 105)
        painter.end()

class 修改密码(QWidget):
    ui=None
    num=1
    def __init__(self):
        super(修改密码, self).__init__()
        self.resize(537, 415)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self._padding = 5
        self._TitleLabel = Title(self)
        self._TitleLabel.setMouseTracking(True)
        self._TitleLabel.setIndent(10)
        self._TitleLabel.move(0, 0)
        self.label = QtWidgets.QLabel(self)
        self.label.setObjectName("label")
        self.label.resize(537, 45)
        self.label.move(0, 0)
        self.label.setStyleSheet("QLabel { background-color: rgb(67,152,253); }")
        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setObjectName("label_2")
        self.label_2.resize(30, 30)
        self.label_2.move(5, 7)
        pix = QPixmap('C:/Users/林逸/Desktop/综合课设三/resource/安全中心.png')
        self.label_2.setPixmap(pix)
        self.label_2.setScaledContents(True)
        self.label_7 = self.label_2 = QtWidgets.QLabel(self)
        self.label_7.setObjectName("label_2")
        self.label_7.resize(200, 30)
        self.label_7.move(35, 7)
        self.label_7.setText('安全中心')
        self.label_7.setStyleSheet("QLabel { color: rgb(255,255,255); }")
        font = QFont()
        font.setFamily("华康方圆体W7")
        font.setPointSize(12)
        self.label_7.setFont(font)
        font = QFont()
        font.setFamily("幼体")
        font.setPointSize(15)
        self.label_3 = QtWidgets.QLabel(self)
        self.label_3.setGeometry(QtCore.QRect(220, 270, 72, 15))
        self.label_3.setObjectName("label_3")
        self.label_3.setText('修改密码')
        self.label_3.setFont(font)
        self.label_3.resize(150, 30)
        self.label_3.move(85, 80)
        font = QFont()
        font.setFamily("楷体")
        font.setPointSize(13)
        self.label_4 = QtWidgets.QLabel(self)
        self.label_4.setGeometry(QtCore.QRect(220, 310, 72, 15))
        self.label_4.setObjectName("label_4")
        self.label_4.setText('请输入账号')
        self.label_4.resize(150, 30)
        self.label_4.move(85, 140)
        self.label_4.setFont(font)
        font1 = QFont()
        font1.setFamily("楷体")
        font1.setPointSize(8)
        self.lab = QtWidgets.QLabel(self)
        self.lab.setGeometry(QtCore.QRect(220, 310, 72, 15))
        self.lab.setObjectName("lab")
        self.lab.setText("")
        self.lab.setStyleSheet("color: red")
        self.lab.resize(180, 30)
        self.lab.move(350, 140)
        self.lab.setFont(font1)
        self.lineEdit_1 = QtWidgets.QLineEdit(self)
        self.lineEdit_1.setGeometry(QtCore.QRect(230, 350, 113, 21))
        self.lineEdit_1.setStyleSheet("background-color: rgb(245,246,250);border-radius: 10px")
        self.lineEdit_1.setObjectName("lineEdit_1")
        self.lineEdit_1.resize(370, 30)
        self.lineEdit_1.move(85, 170)
        self.label_5 = QtWidgets.QLabel(self)
        self.label_5.setGeometry(QtCore.QRect(220, 310, 72, 15))
        self.label_5.setObjectName("label_4")
        self.label_5.setText('请输入密码')
        self.label_5.resize(150, 30)
        self.label_5.move(85, 200)
        self.label_5.setFont(font)
        self.lab2 = QtWidgets.QLabel(self)
        self.lab2.setGeometry(QtCore.QRect(220, 310, 72, 15))
        self.lab2.setObjectName("lab2")
        self.lab2.setText("")
        self.lab2.setStyleSheet("color: red")
        self.lab2.resize(180, 30)
        self.lab2.move(350, 200)
        self.lab2.setFont(font1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self)
        self.lineEdit_2.setGeometry(QtCore.QRect(230, 350, 113, 21))
        self.lineEdit_2.setStyleSheet("background-color: rgb(245,246,250);border-radius: 10px")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_2.resize(370, 30)
        self.lineEdit_2.move(85, 230)

        self.label_6 = QtWidgets.QLabel(self)
        self.label_6.setGeometry(QtCore.QRect(220, 310, 72, 15))
        self.label_6.setObjectName("label_4")
        self.label_6.setText('请再次输入密码')
        self.label_6.resize(180, 30)
        self.label_6.move(85, 260)
        self.label_6.setFont(font)
        self.lab3 = QtWidgets.QLabel(self)
        self.lab3.setGeometry(QtCore.QRect(220, 310, 72, 15))
        self.lab3.setObjectName("lab3")
        self.lab3.setText("")
        self.lab3.setStyleSheet("color: red")
        self.lab3.resize(180, 30)
        self.lab3.move(350, 260)
        self.lab3.setFont(font1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self)
        self.lineEdit_3.setGeometry(QtCore.QRect(230, 350, 113, 21))
        self.lineEdit_3.setStyleSheet("background-color: rgb(245,246,250);border-radius: 10px")
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_3.resize(370, 30)
        self.lineEdit_3.move(85, 290)
        self.lineEdit_2.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_3.setEchoMode(QtWidgets.QLineEdit.Password)
        font = QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        self.lineEdit_1.setFont(font)
        self.lineEdit_2.setFont(font)
        self.lineEdit_3.setFont(font)

        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setGeometry(QtCore.QRect(230, 400, 371, 28))
        self.pushButton.setStyleSheet(
            "QPushButton { background-color: rgb(67,152,253);border-radius: 10px;color: white; }")
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText('确定')
        self.pushButton.resize(370, 40)
        self.pushButton.move(85, 340)

        self._MainLayout = QVBoxLayout()
        self._MainLayout.setSpacing(0)
        self._MainLayout.addWidget(QLabel(), Qt.AlignLeft)
        self._MainLayout.addStretch()
        self.setLayout(self._MainLayout)

        self._CloseButton = TitleButton(b'\xef\x81\xb2'.decode("utf-8"), self)
        self._CloseButton.setObjectName("CloseButton")
        self._CloseButton.setToolTip("关闭窗口")
        self._CloseButton.setMouseTracking(True)
        self._CloseButton.setFixedHeight(self._TitleLabel.height())
        self._CloseButton.clicked.connect(self.close)
        self._MinimumButton = TitleButton(b'\xef\x80\xb0'.decode("utf-8"), self)
        self._MinimumButton.setObjectName("MinMaxButton")
        self._MinimumButton.setToolTip("最小化")
        self._MinimumButton.setMouseTracking(True)
        self._MinimumButton.setFixedHeight(self._TitleLabel.height())
        self._MinimumButton.clicked.connect(self.showMinimized)
        self.setMouseTracking(True)
        self._move_drag = False

    def do(self):
        server = "127.0.0.1"
        user = "sa"
        password = "hbw516626"
        database = "美颜系统"
        conn = pymssql.connect(server, user, password, database, autocommit=True)
        self.cursor = conn.cursor()
        self.cursor.execute('''
SELECT password FROM 用户
WHERE id=\''''+self.lineEdit_1.text()+'\'')
        if self.cursor.fetchall()==[] :
            self.lab.setText('错误账号')
            self.lab2.setText('')
            self.lab3.setText('')
        elif self.lineEdit_2.text()=='':
            self.lab.setText('')
            self.lab2.setText('密码不可为空')
            self.lab3.setText('')
        elif self.lineEdit_2.text()!=self.lineEdit_3.text():
            self.lab.setText('')
            self.lab2.setText('')
            self.lab3.setText('与上个密码不匹配')
        else:
            self.cursor.execute('''
update 用户 set password=%s
where id=%s
''',(self.lineEdit_3.text(),self.lineEdit_1.text()))
            self.cursor.close()
            self.close()
            self.ui = 登录界面()
            self.ui.show()


    # 重写方法以实现拖拽
    def resizeEvent(self, QResizeEvent):
        self._TitleLabel.setFixedWidth(self.width())
        self._CloseButton.move(self.width() - self._CloseButton.width(), 0)
        self._MinimumButton.move(self.width() - (self._CloseButton.width() + 1) * 2 + 1, 0)

    def mousePressEvent(self, event):
        if (event.button() == Qt.LeftButton) and (event.y() < self.label.height()):
            self._move_drag = True
            self.move_DragPosition = event.globalPos() - self.pos()
            event.accept()
        elif (event.button() == Qt.LeftButton) and (event.x() >= 85 and event.x() <= 455) and (
                event.y() >= 170 and event.y() <= 200):
            self.lineEdit_1.setFocus()
            self.num=1
        elif (event.button() == Qt.LeftButton) and (event.x() >= 85 and event.x() <= 455) and (
                event.y() >= 230 and event.y() <= 260):
            self.lineEdit_2.setFocus()
            self.num=2
        elif (event.button() == Qt.LeftButton) and (event.x() >= 85 and event.x() <= 455) and (
                event.y() >= 290 and event.y() <= 320):
            self.lineEdit_3.setFocus()
            self.num=3
        elif (event.button() == Qt.LeftButton) and (event.x() >= 85 and event.x() <= 455) and (
                event.y() >= 340 and event.y() <= 380):
            self.do()

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self._move_drag:
            self.move(QMouseEvent.globalPos() - self.move_DragPosition)
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self._move_drag = False

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.do()
        elif event.key() == Qt.Key_Down:
            if self.num==3:
                self.lineEdit_1.setFocus()
                self.num=1
            elif self.num==1:
                self.lineEdit_2.setFocus()
                self.num = 2
            elif self.num==2:
                self.lineEdit_3.setFocus()
                self.num = 3
        elif event.key() == Qt.Key_Up:
            if self.num == 3:
                self.lineEdit_2.setFocus()
                self.num = 2
            elif self.num == 2:
                self.lineEdit_1.setFocus()
                self.num = 1
            elif self.num == 1:
                self.lineEdit_3.setFocus()
                self.num = 3

    def paintEvent(self, event):
        painter = QPainter()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.begin(self)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawRoundedRect(30, 60, self.width() - 60, self.height() - 80, 20, 20)
        pen = QPen(Qt.gray, 5, Qt.SolidLine)
        pen.setColor(QColor(67, 152, 253))
        painter.setPen(pen)
        # painter.drawLine(115, 255, 425, 255)
        painter.drawLine(80, 85, 80, 105)
        painter.end()

class 登录界面(QWidget):
    change_ui = None
    num=1
    def __init__(self):
        super(登录界面, self).__init__()
        self.resize(537,415)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self._padding = 5
        self._TitleLabel = Title(self)
        self._TitleLabel.setMouseTracking(True)
        self._TitleLabel.setIndent(10)
        self._TitleLabel.move(0, 0)
        self.label = QtWidgets.QLabel(self)
        self.label.setObjectName("label")
        self.label.resize(537, 160)
        self.label.move(0, 0)
        self.title_movie = QMovie('C:/Users/林逸/Desktop/综合课设三/resource/2.gif')
        self.title_movie.setScaledSize(QtCore.QSize(537, 160))
        self.label.setMovie(self.title_movie)
        self.title_movie.start()
        self.label.setWindowFlags(Qt.WindowStaysOnBottomHint)

        self.label2 = QtWidgets.QLabel(self)
        self.label2.setObjectName("label2")
        self.label2.resize(50, 50)
        self.label2.move(10, 10)
        self.title_pix = QPixmap('C:/Users/林逸/Desktop/综合课设三/resource/卡通人像.png')
        self.label2.setPixmap(self.title_pix)
        self.label2.setScaledContents(True)

        self.label3 = QtWidgets.QLabel(self)
        self.label3.setObjectName("label3")
        self.label3.resize(60, 50)
        self.label3.move(60, 10)
        self.label3.setText('塑颜')
        self.label3.setStyleSheet("QLabel{color:rgb(255,255,255,255);border: 0px;font: 18pt \"华康方圆体W7\";}")
        self.label3.setAlignment(Qt.AlignVCenter)

        self.label4 = QtWidgets.QLabel(self)
        self.label4.setObjectName("label4")
        self.label4.resize(80, 80)
        self.label4.move(228, 120)
        pix = QPixmap('C:/Users/林逸/Desktop/综合课设三/resource/头像.png')
        painter = QPainter(pix)
        painter.begin(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        path = QPainterPath()
        path.addEllipse(0, 0, 200, 200)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, 200, 200, pix)
        painter.end()
        self.label4.setPixmap(pix)
        self.label4.setScaledContents(True)

        self.lab = QtWidgets.QLabel(self)
        self.lab.setObjectName("lab")
        self.lab.move(115, 225)
        self.lab.resize(25, 25)
        pix2=QPixmap('C:/Users/林逸/Desktop/综合课设三/resource/账号.png')
        self.lab.setPixmap(pix2)
        self.lab.setScaledContents(True)
        self.lineEdit_1 = QtWidgets.QLineEdit(self)
        self.lineEdit_1.setObjectName("lineEdit_1")
        self.lineEdit_1.setStyleSheet("border: none;")
        font = QFont()
        font.setFamily("宋体")
        font.setPointSize(15)
        self.lineEdit_1.setFont(font)
        self.lineEdit_1.move(147, 223)
        self.lineEdit_1.resize(277, 30)

        self.lab_2 = QtWidgets.QLabel(self)
        self.lab_2.setObjectName("lab_2")
        self.lab_2.move(115, 270)
        self.lab_2.resize(25, 25)
        pix3 = QPixmap('C:/Users/林逸/Desktop/综合课设三/resource/账号安全.png')
        self.lab_2.setPixmap(pix3)
        self.lab_2.setScaledContents(True)
        self.lineEdit_2 = QtWidgets.QLineEdit(self)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_2.setStyleSheet("border: none;")
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_2.move(147, 268)
        self.lineEdit_2.resize(277, 30)

        self.checkBox = QtWidgets.QCheckBox(self)
        self.checkBox.setObjectName("checkBox")
        self.checkBox.setText('显示密码')
        self.checkBox.setChecked(False)
        self.checkBox.setStyleSheet("QCheckBox::indicator {color: rgb(204, 204, 204);}")
        self.checkBox.move(117,310)
        self.checkBox.resize(100,25)
        self.checkBox.toggled.connect(self.showpassword)
        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText('注册')
        self.pushButton.move(220,310)
        self.pushButton.resize(100,25)
        self.pushButton.setStyleSheet("QPushButton { background-color: transparent; }")
        self.pushButton_2 = QtWidgets.QPushButton(self)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setText('修改密码')
        self.pushButton_2.move(345, 310)
        self.pushButton_2.resize(100, 25)
        self.pushButton_2.setStyleSheet("QPushButton { background-color: transparent; }")
        self.pushButton_3 = QtWidgets.QPushButton(self)
        self.pushButton_3.setGeometry(QtCore.QRect(520, 370, 93, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setText('登录')
        self.pushButton_3.move(123, 348)
        self.pushButton_3.resize(297,40)
        self.pushButton_3.setStyleSheet("QPushButton { background-color: rgb(7,188,252);border-radius: 10px;color: white; }")

        self._MainLayout = QVBoxLayout()
        self._MainLayout.setSpacing(0)
        self._MainLayout.addWidget(QLabel(), Qt.AlignLeft)
        self._MainLayout.addStretch()
        self.setLayout(self._MainLayout)

        self._CloseButton = TitleButton(b'\xef\x81\xb2'.decode("utf-8"), self)
        self._CloseButton.setObjectName("CloseButton")
        self._CloseButton.setToolTip("关闭窗口")
        self._CloseButton.setMouseTracking(True)
        self._CloseButton.setFixedHeight(self._TitleLabel.height())
        self._CloseButton.clicked.connect(self.close)
        self._MinimumButton = TitleButton(b'\xef\x80\xb0'.decode("utf-8"), self)
        self._MinimumButton.setObjectName("MinMaxButton")
        self._MinimumButton.setToolTip("最小化")
        self._MinimumButton.setMouseTracking(True)
        self._MinimumButton.setFixedHeight(self._TitleLabel.height())
        self._MinimumButton.clicked.connect(self.showMinimized)
        self.setMouseTracking(True)
        self._move_drag = False


    def showpassword(self):
        state = self.checkBox.checkState()
        if state == Qt.Checked:
            self.lineEdit_2.setEchoMode(QtWidgets.QLineEdit.Normal)
        else:
            self.lineEdit_2.setEchoMode(QtWidgets.QLineEdit.Password)

    def do(self):
        server = "127.0.0.1"
        user = "sa"
        password = "hbw516626"
        database = "美颜系统"
        conn = pymssql.connect(server, user, password, database, autocommit=True)
        self.cursor = conn.cursor()
        self.cursor.execute('''
SELECT password FROM 用户
WHERE id=\'''' + self.lineEdit_1.text() + '\'')
        item=self.cursor.fetchall()
        if item == []:
            rect = RECT()
            current_win = win32gui.GetForegroundWindow()  # 获取当前窗口
            ctypes.windll.user32.GetWindowRect(current_win, ctypes.byref(rect))  # 获取当前窗口坐标
            for i in range(2, 30):
                win32gui.SetWindowPos(current_win, None, rect.left + 5 * random.randint(1, i),rect.top - 5 * random.randint(1, i), rect.right - rect.left,rect.bottom - rect.top, win32con.SWP_NOSENDCHANGING | win32con.SWP_SHOWWINDOW)  # 实现更改当前窗口位置
            win32gui.SetWindowPos(current_win, None, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top,win32con.SWP_NOSENDCHANGING | win32con.SWP_SHOWWINDOW)   #恢复到初始位置
        else:
            if item[0][0]==self.lineEdit_2.text():
                self.close()
                self.ui=UI(self.lineEdit_1.text())
                self.ui.show()
            else:
                rect = RECT()
                current_win = win32gui.GetForegroundWindow()  # 获取当前窗口
                ctypes.windll.user32.GetWindowRect(current_win, ctypes.byref(rect))  # 获取当前窗口坐标
                for i in range(2, 30):
                    win32gui.SetWindowPos(current_win, None, rect.left + 5 * random.randint(1, i),
                                          rect.top - 5 * random.randint(1, i), rect.right - rect.left,
                                          rect.bottom - rect.top,
                                          win32con.SWP_NOSENDCHANGING | win32con.SWP_SHOWWINDOW)  # 实现更改当前窗口位置
                win32gui.SetWindowPos(current_win, None, rect.left, rect.top, rect.right - rect.left,
                                      rect.bottom - rect.top, win32con.SWP_NOSENDCHANGING | win32con.SWP_SHOWWINDOW)
    def now(self):
        self.close()
        self.now_ui = 注册()
        self.now_ui.show()

    def change(self):
        self.close()
        self.change_ui=修改密码()
        self.change_ui.show()
    # 重写方法以实现拖拽
    def resizeEvent(self, QResizeEvent):
        self._TitleLabel.setFixedWidth(self.width())
        self._CloseButton.move(self.width() - self._CloseButton.width(), 0)
        self._MinimumButton.move(self.width() - (self._CloseButton.width() + 1) * 2 + 1, 0)
    def mousePressEvent(self, event):
        if (event.button() == Qt.LeftButton) and (event.y() < self._TitleLabel.height()):
            self._move_drag = True
            self.move_DragPosition = event.globalPos() - self.pos()
            event.accept()
        elif (event.button() == Qt.LeftButton) and (event.x()>=147 and event.x()<=424) and (event.y()>=223 and event.y()<=253):
            self.lineEdit_1.setFocus()
            self.num=1
        elif (event.button() == Qt.LeftButton) and (event.x()>=147 and event.x()<=424) and (event.y()>=268 and event.y()<=298):
            self.lineEdit_2.setFocus()
            self.num=2
        elif (event.button() == Qt.LeftButton) and (event.x()>=self.checkBox.pos().x() and event.x()<=self.checkBox.pos().x()+self.checkBox.size().width()-20) and (event.y()>=self.checkBox.pos().y() and event.y()<=self.checkBox.pos().y()+self.checkBox.size().height()):
            state = self.checkBox.checkState()
            if state == Qt.Checked:
                self.checkBox.setCheckState(Qt.Unchecked)
            else:
                self.checkBox.setCheckState(Qt.Checked)
        elif (event.button() == Qt.LeftButton) and (event.x() >= self.pushButton.pos().x() and event.x() <= self.pushButton.pos().x() + self.pushButton.size().width() - 20) and (event.y() >= self.pushButton.pos().y() and event.y() <= self.pushButton.pos().y() + self.pushButton.size().height()):
            self.now()
        elif (event.button() == Qt.LeftButton) and (event.x() >= self.pushButton_2.pos().x() and event.x() <= self.pushButton_2.pos().x() + self.pushButton_2.size().width() - 20) and (event.y() >= self.pushButton_2.pos().y() and event.y() <= self.pushButton_2.pos().y() + self.pushButton_2.size().height()):
            self.change()
        elif (event.button() == Qt.LeftButton) and (event.x() >= self.pushButton_3.pos().x() and event.x() <= self.pushButton_3.pos().x() + self.pushButton_3.size().width() - 20) and (event.y() >= self.pushButton_3.pos().y() and event.y() <= self.pushButton_3.pos().y() + self.pushButton_3.size().height()):
            self.do()
    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self._move_drag:
            self.move(QMouseEvent.globalPos() - self.move_DragPosition)
            QMouseEvent.accept()
    def mouseReleaseEvent(self, QMouseEvent):
        self._move_drag = False
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.do()
        elif event.key() == Qt.Key_Down:
            if self.num==2:
                self.lineEdit_1.setFocus()
                self.num=1
            else:
                self.lineEdit_2.setFocus()
                self.num=2
        elif event.key() == Qt.Key_Up:
            if self.num==2:
                self.lineEdit_1.setFocus()
                self.num=1
            else:
                self.lineEdit_2.setFocus()
                self.num=2


    
    def paintEvent(self, event):
        painter = QPainter()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.begin(self)
        painter.fillRect(0, 0, self.width(), self.height(), QColor(255, 255, 255))
        pen = QPen(Qt.gray, 0, Qt.SolidLine)
        pen.setColor(QColor(234,234,234))
        painter.setPen(pen)
        painter.drawLine(115, 255, 425, 255)
        painter.drawLine(115,300,425,300)
        painter.end()

if __name__ =='__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(open("C:/Users/林逸/Desktop/综合课设三/style.qss").read())
    ui = 登录界面()
    ui.show()
    sys.exit(app.exec_())