'''
HW2024 Radar Station main program
'''

# 5.2 本版本适用于中间的mindvision和两个海康相机，两个海康相机用长焦专门看飞坡和对面高地
# 5.16 三个相机如果还开启录像的话帧率实在太低
# 6.30 分析比赛视频，其实感觉左边再加一个相机足够了，右边因为遮挡的问题加了相机和中间相机其实效果很接近的，即中间相机可以识别到对面吊射的英雄，考虑到帧率，没问题
# 在本版本的代码中，相机1代表左边相机

import cv2
import numpy as np
import time
import traceback  # 打印栈信息
import sys       # 系统操作模块
import os       # 与系统交互
import threading   
import pickle as pkl  
from datetime import datetime
import logging   # 输出日志
import queue   # 线程同步的队列类

from radar_module.camera import read_yaml, white_balance, Camera_Thread
from radar_module.location import locate_pick, locate_record
from radar_module.Lidar import Radar, DepthQueue
from radar_module.network import Predictor
from radar_module.common import is_inside, draw_alarm_area, point_3D_to_2D_coor_4points, \
    draw_car_circle, world_coord_2_map_img_coord, draw_alarm_area_map, xyCoorCompensator,\
    draw_3d_point_in_uv, get_roi_img, middle_record, left_record
from radar_module.config import reverse, debug_info, alarm_region, T_w2c_0, T_c2w_0, T_w2c_1, T_c2w_1, T_w2c_2, T_c2w_2, red, blue, \
    alliance_game, USB, using_serial, using_video, using_dq, home_test, loc2priority, loc2car, color2enemy,\
    Car_Info, Our_Car_Info, uv_alarm, coor_alarm, demo,\
    MAP_PATH, PC_RECORD_SAVE_DIR, VIDEO_PATH, PC_RECORD_FRAMES, NET_PATH_CAR, NET_PATH_NUMBER, YAML_PATH_CAR, YAML_PATH_NUMBER, PASS_WORD,\
    ARER_LENGTH, AREA_WIDTH, CAM_RESTART_TIME_INTERVAL, LOG_DIR, twoCam, thirdCam, whiteBalance, record_fps,\
    CAM_CENTER_L, CAM_CENTER_R, CAM_CENTER_BASE, ROI_SIZE, ROI_BASE_SIZE, CAM_CENTER_BASE_VIEW, ROI_BASE_VIEW_SIZE, roi,\
    net_half, net_size_car, net_size_number, open_track

from ui import Mywindow, HP_scene
import ui
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage,QPixmap

import serial
import _thread # 低级别的多线程，与threading配合
import pexpect   # 自动问答
from UART import UART_passer, read, write  # 串口操作

# 定时器任务 
def heart_beat():
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    threading.Timer(1, heart_beat).start() # 1s定时器

# 裁判系统发送函数
def send_judge(message: dict, QTwindow: Mywindow):
    '''
    alarming message dick format define:
    {'task': 2, 'data': [loc, alarm_target, team]} loc:”预警区域“  alarm_target: 在预警区域内的车辆编号列表  team:R or B
    {'task': 3, 'data': [alarm_target]}

    ps: alarm_target : a list of the car inside the region
    '''

    # 这里把信息都包装成了list，在这里序号从1开始，之后uart部分从0开始
    if message['task'] == 1: # 发送敌方车辆坐标
        loc = []
        whole_location: dict = message['data'][0]
        for i in range(1, 7):
            key = str(i)
            coor = whole_location[key][:2] 
            loc.append(coor)
        UART_passer.push_loc(loc) # 放入当前帧位置

    if message['task'] == 2: # 预警
        loc, alarm_target, team = message['data']
        UART_passer.push(loc2priority[loc], loc2car[loc], alarm_target, color2enemy[team]) # 将一个预警push进预警队列，以loc作为优先级，编号越小，优先级越高

    if message['task'] == 3: # 己方车辆坐标
        our_loc = []
        our_whole_location: dict = message['data'][0]
        for i in range(1, 7):
            key = str(i)
            our_coor = our_whole_location[key][:2]
            our_loc.append(our_coor)
        UART_passer.push_our_loc(our_loc) # 发送逻辑未写

# 主程序类
class radar_process(object):
    def __init__(self, QTwindow: Mywindow):

        self._myshow = QTwindow
        self._frame_num = 0
        # self._fps_queue = queue.Queue(30)
        self._cv_window_init = False # 需要初始化cv2window
        self._radar = []  
        self._position_flag = False  # 未完成位姿标定，只针对中间相机
        self._K_0 = []
        self._imgsz = []
        self._E_0 = []   # 5.6 方便后面外参转换
         
        # 初始化相机内外参
        # 现在有三个相机，中间mindvision为0
        # 面向前方，livox右边为1，左边为2
        
        # 小地图初始化，我方朝下
        self._map_img = cv2.imread(MAP_PATH) # tiny map
        if reverse: # 我方B
            self._map_img = cv2.rotate(self._map_img,cv2.ROTATE_90_CLOCKWISE)
        else: # 我方R
            self._map_img = cv2.rotate(self._map_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 雷达初始化
        
        # 5.3 相机之间间隔比较大，双目标定有些困难，所以配置参数都是统一的和雷达的外参
        # 相机0
        _, K_0, C_0, E_0, imgsz = read_yaml(0) # camera0 info
        self._imgsz.append(imgsz)  
        self._K_0.append(K_0)
        self._E_0.append(E_0)
        if using_dq: # 使用点云队列
            self._radar.append(DepthQueue(10, imgsz, K_0, C_0, E_0))
            # 填入点云
            with open(PC_RECORD_SAVE_DIR, 'rb') as f:
                radar_frames = pkl.load(f)  # 4.17 点云肯定都是一样的
            for frame_r in radar_frames:
                self._radar[0].push_back(frame_r)
        else:
            self._radar.append(Radar(K_0, C_0, E_0, queue_size = 10, imgsz=imgsz))
        
        # 相机1
        _, K_0, C_0, E_0, imgsz = read_yaml(1) # camera1 info
        self._imgsz.append(imgsz)
        self._K_0.append(K_0)
        self._E_0.append(E_0)
        if using_dq: # 使用点云队列
            self._radar.append(DepthQueue(10, imgsz, K_0, C_0, E_0))
            # 填入点云
            with open(PC_RECORD_SAVE_DIR, 'rb') as f:
                radar_frames = pkl.load(f)  # 4.17 点云肯定都是一样的
            for frame_r in radar_frames:
                self._radar[1].push_back(frame_r)
        else:
            self._radar.append(Radar(K_0, C_0, E_0, queue_size = 10, imgsz=imgsz))
            
        # 相机2
        # _, K_0, C_0, E_0, imgsz = read_yaml(2) # camera1 info
        # self._imgsz.append(imgsz)
        # self._K_0.append(K_0)
        # self._E_0.append(E_0)
        # if using_dq: # 使用点云队列
        #     self._radar.append(DepthQueue(10, imgsz, K_0, C_0, E_0))
        #     # 填入点云
        #     with open(PC_RECORD_SAVE_DIR, 'rb') as f:
        #         radar_frames = pkl.load(f)  # 4.17 点云肯定都是一样的
        #     for frame_r in radar_frames:
        #         self._radar[2].push_back(frame_r)
        # else:
        #     self._radar.append(Radar(K_0, C_0, E_0, queue_size = 10, imgsz=imgsz))
        
        if not using_dq:
            Radar.start() # 雷达开始工作
        
        # 雷达队列初始化标志
        if using_dq:
            self._radar_init = [True,True] # 当使用点云队列测试时，直接默认为True
        else:
            self._radar_init = [False,False]  # 应该是后面spin_once再改

        # YOLOV8
        # 神经网络初始化，predictor类使用现有权重文件
        self._net = Predictor(weights_CAR=NET_PATH_CAR, classes_CAR=YAML_PATH_CAR, weights_NUMBER=NET_PATH_NUMBER, classes_NUMBER=YAML_PATH_NUMBER)
        logger.info('net init finish')

        # open main process camera
        # 相机线程开始，并可自由调节相机数量
        if using_video:
            self._cap0 = Camera_Thread(0, camera_brand = 0, video = True, video_path = VIDEO_PATH)
            self._cap1 = Camera_Thread(1, camera_brand = 1, video = True, video_path = VIDEO_PATH)
            # self._cap2 = Camera_Thread(2, camera_brand = 1, video = True, video_path = VIDEO_PATH)
        else:
            self._cap0 = Camera_Thread(0, camera_brand = 0)
            self._cap1 = Camera_Thread(1, camera_brand = 1) 
            # self._cap2 = Camera_Thread(2, camera_brand = 1)
        if self._cap0.is_open():
            logger.info("Camera {0} Starting.".format(0))
        else:
            logger.warning("Camera {0} Failed, try to open.".format(0))
        if self._cap1.is_open():
            logger.info("Camera {0} Starting.".format(1))
        else:
            logger.warning("Camera {0} Failed, try to open.".format(1))
        # if self._cap2.is_open():
        #     logger.info("Camera {0} Starting.".format(2))
        # else:
        #     logger.warning("Camera {0} Failed, try to open.".format(2))

        # opencv window
        # using_dq为真表示使用录制点云不使用节点
        if not using_dq:
            cv2.namedWindow("depth",cv2.WINDOW_NORMAL) # 雷达深度图
        else: # 使用点云队列
            cv2.namedWindow("result_img",cv2.WINDOW_NORMAL) # 结果
        if debug_info:        # 是否进入调试模式
            pass
            # cv2.namedWindow("frame0",cv2.WINDOW_NORMAL) # 相机原始图像
            # # cv2.namedWindow("yolo_result_img",cv2.WINDOW_NORMAL) # YOLOV5预测结果
            # cv2.namedWindow("frame1",cv2.WINDOW_NORMAL) # 相机原始图像
            # cv2.namedWindow("frame2",cv2.WINDOW_NORMAL) # 相机原始图像
            # if roi:     # 其实只有一个相机，好像是把一个画面切成两半
            #     cv2.namedWindow("ROI_L",cv2.WINDOW_NORMAL) # 左前哨站
            #     cv2.namedWindow("ROI_R",cv2.WINDOW_NORMAL) # 右前哨站
            #     cv2.namedWindow("ROI_BASE",cv2.WINDOW_NORMAL) # 基地截取

        self.get_position_using_last() # 尝试读取之前保存的位姿
        self.get_other_two_cam_position()  # 利用相对外参估计左右相机的位姿
        self._base_open = False # 基地视角

        # 定时重启掉线相机
        self._retray_flag = False
        self._last_time = time.time()   
        self._retray_flag0 = False
        self._last_time0 = time.time()
        self._retray_flag2 = False
        self._last_time2 = time.time()
    
    # 5.3 只有中间的相机做位姿标定，左右相机直接计算得出位姿
    def get_position_using_last(self):
        '''
        使用保存的位姿
        '''
        # 尝试读取之前保存的位姿

        # cam 0
        flag0 = False
        flag0, rvec0, tvec0 = locate_record(camera_type = 0, reverse= reverse, save = False, rvec = None, tvec = None)
        if flag0:
            global T_w2c_0
            global T_c2w_0
            T_w2c_0 = np.eye(4)
            T_w2c_0[:3, :3] = cv2.Rodrigues(rvec0)[0]   # 旋转向量与旋转矩阵之间的转换
            T_w2c_0[:3, 3] = tvec0.reshape(-1)
            T_c2w_0 = np.linalg.inv(T_w2c_0)
            logger.info("中间相机读取现有位姿成功")
            QTwindow.locate_pick_state = True
            QTwindow.locate_pick_start = False
            QTwindow.btn1.setText("中间相机标定新位姿")
            QTwindow.set_text("feedback","中间相机读取现有位姿成功")
            self._position_flag = True
            # 将位姿存入反投影预警类
            # T, cp = self._scene[0].push_T(rvec0, tvec0)
            # 将位姿存入位置预警类
            # self._alarm_map.push_T( T, cp, 0)
        else:
            logger.info("中间相机读取现有位姿失败")
            QTwindow.set_text("feedback", "中间相机读取现有位姿失败")
            self._position_flag = False

 
        # 新的位姿标定
        # 4.19 只要读取到上一次的位姿或者新标成功了新的位姿，position_flag就是True
    def get_position_new(self):
        '''
        using huge range object to get position, which is simple but perfect
        '''
        
        # 4.19 在初始化阶段其实就已经读取了已有位姿
        # 注意这里就算没调到对应相机视角也不用管

        # cam0
        flag0 = False
        if self._cap0.is_open():
            logger.info("中间相机开始位姿标定")
            # 4.19 该函数会等待操作完成后返回，标定失败的话保留初始化时读取的位姿
            flag0, rvec0, tvec0 = locate_pick(self._cap0, reverse, 0,  home_size= home_test, video_test=using_video)
        if flag0:
            global T_w2c_0
            global T_c2w_0
            T_w2c_0 = np.eye(4)
            T_w2c_0[:3, :3] = cv2.Rodrigues(rvec0)[0]
            T_w2c_0[:3, 3] = tvec0.reshape(-1)
            T_c2w_0 = np.linalg.inv(T_w2c_0)
            locate_record(0, reverse, True, rvec0, tvec0) # 保存位姿
            QTwindow.locate_pick_state = True
            QTwindow.locate_pick_start = False
            QTwindow.btn1.setText("中间相机再次标定位姿")
            QTwindow.set_text("feedback","位姿标定完成")
            logger.info("中间相机位姿标定完成")
            self._position_flag = True
            # T, cp = self._scene[0].push_T(rvec, tvec)
            # self._alarm_map.push_T( T, cp, 0)
        else:
            QTwindow.locate_pick_start = False
            QTwindow.btn1.setText("再次标定中间相机位姿")
            QTwindow.set_text("feedback", "Camera 0 pose init error")
            logger.info("Camera 0 pose init error")
        
        # 如果使用视频，将其恢复至视频开始
        if using_video:
            self._cap0.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pass
    
    # 坐标系之间的转换
    # 5.5 16mm镜头还没到，现在还转换不了，先都把位姿设成一样的
    # 一定注意global ! ! !
    # 只要中间相机读取到了之前的位姿文件或者新标定过了，那么另外两个相机都能确定，所以不需要保存这两个相机的位姿
    # 思路是推出雷达位姿，然后变换到另外两个相机
    def get_other_two_cam_position(self):
        if self._position_flag:
            global T_w2c_1
            global T_c2w_1
            # global T_w2c_2
            # global T_c2w_2
            T_w2c_1 = T_w2c_0 @ (np.linalg.inv(self._E_0[0])) @ self._E_0[1]
            T_c2w_1 = np.linalg.inv(T_w2c_1)
            # T_w2c_2 = T_w2c_0 @ (np.linalg.inv(self._E_0[0])) @ self._E_0[2]
            # T_c2w_2 = np.linalg.inv(T_w2c_2)

    def spin_once(self):
        '''
        雷达站主程序的一个循环
        '''
        t_spin_once= time.time()
        self._frame_num += 1

        # 定时重启掉线相机
        if (t_spin_once-self._last_time0 > CAM_RESTART_TIME_INTERVAL):    # 多少时间间隔重启
            self._retray_flag0 = True
            self._last_time0 = t_spin_once
        if (t_spin_once-self._last_time > CAM_RESTART_TIME_INTERVAL):
            self._retray_flag = True
            self._last_time = t_spin_once
        # if (t_spin_once-self._last_time2 > CAM_RESTART_TIME_INTERVAL):
        #     self._retray_flag2 = True
        #     self._last_time2 = t_spin_once
        

        # check radar init
        # 5.5 现在有三个相机，相当于三个雷达类管理
        if not using_dq:
            if self._radar[0].check_radar_init():
                self._myshow.set_text("feedback", "radar of camera 0 init")
                self._radar_init = True
                logger.info("radar of camera 0 init")
            if self._radar[1].check_radar_init():
                self._myshow.set_text("feedback", "radar of camera 1 init")
                self._radar_init = True
                logger.info("radar of camera 1 init")
            # if self._radar[2].check_radar_init():
            #     self._myshow.set_text("feedback", "radar of camera 2 init")
            #     self._radar_init = True
            #     logger.info("radar of camera 2 init")
            depth = self._radar[0].read()# 获得lidar深度图，随便哪个都行，这个只是拿来调试的
        
        # get image and do prediction
        if not self._cap0.is_open():
            if self._retray_flag0:
                self._retray_flag0 = False
                self._cap0.open()
                logger.error("相机0掉线")
        
        if not self._cap1.is_open():
            if self._retray_flag :
                self._retray_flag = False
                self._cap1.open()
                logger.error("相机1掉线")
                
        # if not self._cap2.is_open():
        #     if self._retray_flag :
        #         self._retray_flag = False
        #         self._cap2.open()
        #         logger.error("相机2掉线")

        t6 = time.time()
        self._cap0._lock.acquire()
        flag0, frame0 = self._cap0.read() # 9ms
        self._cap0._lock.release()
        
        self._cap1._lock.acquire()
        flag1, frame1 = self._cap1.read()
        self._cap1._lock.release()
        
        # self._cap2._lock.acquire()
        # flag2, frame2 = self._cap2.read()
        # self._cap2._lock.release()
        
        t7 = time.time()
        print("get image time is:{0}".format(t7-t6))  # 5.12 取图时间0.025s左右
        
        if not flag0:
            self._myshow.set_text("feedback", "CAMERA 0 BROKEN waiting until they resume")
            # time.sleep(0.01)
            # return
        if not flag1:
            self._myshow.set_text("feedback", "CAMERA 1 BROKEN waiting until they resume")
        # if not flag2:
        #     self._myshow.set_text("feedback", "CAMERA 2 BROKEN waiting until they resume")
        
        # 至少要一个相机能用
        if flag0 or flag1:
            
            # 5.3 左右相机不白平衡，后续如果太慢就去掉yolo_result_img
            if flag0:
                if whiteBalance:
                    frame0 = white_balance(frame0) # 白平衡，耗时较长
                yolo_result_img_0 = frame0.copy() # yolo输出raw结果
                result_img_0 = frame0.copy() # 去重输出结果
            
            if flag1:
                yolo_result_img_1 = frame1.copy() # yolo输出raw结果
                result_img_1 = frame1.copy() # 去重输出结果
            
            # if flag2:
            #     yolo_result_img_2 = frame2.copy() # yolo输出raw结果
            #     result_img_2 = frame2.copy() # 去重输出结果
            
            # UI操作
            # UI按键标定位姿
            if QTwindow.locate_pick_start: # 按键按下用四点手动标定估计位姿
                self.get_position_new()
                
            self.get_other_two_cam_position()
            
            # UI按键录像    
            # 6.1 暂时三个相机同时开始录制
            if QTwindow.record_state[0] and QTwindow.record_state[1]:
                if ui.open_thread:
                    try:
                        _thread.start_new_thread(middle_record, (self._cap0,))
                    except:
                        print("open middle record thread failed!")
                        
                    try:
                        _thread.start_new_thread(left_record, (self._cap1,))
                    except:
                        print("open left record thread failed!")
                        
                    # try:
                    #     _thread.start_new_thread(right_record, (self._cap1,))
                    # except:
                    #     print("open right record thread failed!")
                    
                    ui.open_thread = False
            
            # 4.19 该部分功能暂时去除
            # UI打开关闭基地视角
            # if QTwindow.base_view: # 按键按下，基地画面最大化
            #     self._base_open = True
            # else:
            #     self._base_open = False
            # if self._base_open and self._cv_window_init:
            #     cv2.namedWindow("base_view",cv2.WINDOW_AUTOSIZE) # 全屏后的基地视角
            #     # cv2.resizeWindow("base_view", ROI_BASE_VIEW_SIZE[0], ROI_BASE_VIEW_SIZE[1]) # time costs
            #     cv2.moveWindow("base_view",630,0)
            #     self._cv_window_init = False
            # elif self._base_open == False:
            #     # cv2.destroyWindow("base_view")
            #     self._cv_window_init = True

  
            # 遍历预警区域进行可视化
            # 5.3 只有中间相机有足够的视野看到现在的预警区域
            # 5.13 为了评估左右相机现场标定的效果，在其上分别只绘制一个区域
            map_img = self._map_img.copy()
            if self._position_flag:
                for key,value in alarm_region.items():
                    point_3D_L_T = alarm_region[key][0]
                    point_3D_L_B = alarm_region[key][1]
                    point_3D_R_B = alarm_region[key][2]
                    point_3D_R_T = alarm_region[key][3]
                    draw_alarm_area(point_3D_L_T, point_3D_L_B, point_3D_R_B, point_3D_R_T, T_w2c_0, self._K_0[0], result_img_0, 2) # main_demo中可视化预警区域
                    draw_alarm_area_map(point_3D_L_T, point_3D_L_B, point_3D_R_B, point_3D_R_T, map_img, 2) # 小地图中可视化预警区域
                    
                    # 5.13
                    # if key == 'alarm_area_dafu':
                    #     draw_alarm_area(point_3D_L_T, point_3D_L_B, point_3D_R_B, point_3D_R_T, T_w2c_2, self._K_0[2], result_img_2, 2) # 左相机
            
            t8 = time.time()
            print("copy image and draw alaem: ".format(t8-t7))
            # 神经网络预测 
            # 去重的话，上交代码里神经网络部分没有开源，两个相机还用单相机那种去重肯定不行
            # 考虑改写现在的匹配函数，直接预测的时候赋予标签，并保留是哪个相机预测的，因为后面解算坐标需要对应
            
            # 5.13 明天优先录制视频
            t1 = time.time()
            
            if flag0:
                self._net.infer(frame0, camera_type = 0, open_track = open_track)

            if flag1:
                # pass
                self._net.infer(frame1, camera_type = 1, open_track = open_track)
                
            # if flag2:
            #     # pass
            #     self._net.infer(frame2, camera_type = 2, open_track = open_track)
                
            t2 = time.time()
            print("infer time is: {0}".format(t2-t1))
            
            # 4.19 修改去重规则，在该函数内部完成两个相机的去重
            car_number, FinalResult = self._net.network_result_process()  # 返回各个车辆和装甲板的boundingbox
            
            t3 = time.time()
            print("quchong time is: {0}".format(t3-t2))
          

            # 每一帧数据复位（暂不采用小地图记忆模式，每帧更新坐标与预警信息）
            # 4.19 左右相机用不同的相机外参，但是最终整合到一起的数据结构不变
            for num in range(7):
                Car_Info[num].id = num
                Car_Info[num].x = 0
                Car_Info[num].y = 0
                Car_Info[num].z = 0
                Car_Info[num].alarm = False
                Car_Info[num].area = 0
                Our_Car_Info[num].id = num
                Our_Car_Info[num].x = 0
                Our_Car_Info[num].y = 0
                Our_Car_Info[num].z = 0

            # 预测结果后处理与可视化-car
            if car_number > 0:
                for i in range(car_number):
                    camera_id = FinalResult[i].camera_type
                    # 可视化车辆 bbox
                    if FinalResult[i].color == red:
                        bbx_color = (0, 0, 255)
                        id_string = 'R' + str(FinalResult[i].num)
                        txt_color = (255, 255, 255)
                    elif FinalResult[i].color == blue:
                        bbx_color = (255, 0, 0)
                        id_string = 'B' + str(FinalResult[i].num)
                        txt_color = (255, 255, 255)
                    else:
                        bbx_color = (100, 100, 100)
                        id_string = ''
                        txt_color = (255, 255, 255)
                    
                    # 4.28 在对应相机画面绘制预测结果
                    # 5.6 比赛的时候感觉可以把demo关了
                    if demo:
                        if QTwindow.camera0_view and camera_id == 0:
                            cv2.rectangle(result_img_0, (int(FinalResult[i].x1), int(FinalResult[i].y1)), (int(FinalResult[i].x2), int(FinalResult[i].y2)), bbx_color, 4) # car bbox
                            cv2.rectangle(result_img_0, (int(FinalResult[i].armor_x1), int(FinalResult[i].armor_y1)), (int(FinalResult[i].armor_x2), int(FinalResult[i].armor_y2)), bbx_color, 4) # armor bbox
                            cv2.rectangle(result_img_0, (int(FinalResult[i].x1), int(FinalResult[i].y1)), (int(FinalResult[i].x1)+90, int(FinalResult[i].y1)-60), bbx_color, -1) # text background
                            cv2.putText(result_img_0, id_string, (int(FinalResult[i].x1), int(FinalResult[i].y1)-10), cv2.FONT_HERSHEY_COMPLEX, 2, txt_color, 2) # text

                        elif QTwindow.camera1_view and camera_id == 1:
                            cv2.rectangle(result_img_1, (int(FinalResult[i].x1), int(FinalResult[i].y1)), (int(FinalResult[i].x2), int(FinalResult[i].y2)), bbx_color, 4) # car bbox
                            cv2.rectangle(result_img_1, (int(FinalResult[i].armor_x1), int(FinalResult[i].armor_y1)), (int(FinalResult[i].armor_x2), int(FinalResult[i].armor_y2)), bbx_color, 4) # armor bbox
                            cv2.rectangle(result_img_1, (int(FinalResult[i].x1), int(FinalResult[i].y1)), (int(FinalResult[i].x1)+90, int(FinalResult[i].y1)-60), bbx_color, -1) # text background
                            cv2.putText(result_img_1, id_string, (int(FinalResult[i].x1), int(FinalResult[i].y1)-10), cv2.FONT_HERSHEY_COMPLEX, 2, txt_color, 2) # text
                        
                        # elif QTwindow.camera2_view and camera_id == 2:
                        #     cv2.rectangle(result_img_2, (int(FinalResult[i].x1), int(FinalResult[i].y1)), (int(FinalResult[i].x2), int(FinalResult[i].y2)), bbx_color, 4) # car bbox
                        #     cv2.rectangle(result_img_2, (int(FinalResult[i].armor_x1), int(FinalResult[i].armor_y1)), (int(FinalResult[i].armor_x2), int(FinalResult[i].armor_y2)), bbx_color, 4) # armor bbox
                        #     cv2.rectangle(result_img_2, (int(FinalResult[i].x1), int(FinalResult[i].y1)), (int(FinalResult[i].x1)+90, int(FinalResult[i].y1)-60), bbx_color, -1) # text background
                        #     cv2.putText(result_img_2, id_string, (int(FinalResult[i].x1), int(FinalResult[i].y1)-10), cv2.FONT_HERSHEY_COMPLEX, 2, txt_color, 2) # text
                    
                    
                    # 计算世界坐标系下目标坐标
                    # 5.2 只要中间相机搞好了就行
                    if self._position_flag:
                        if self._radar_init:
                            rect = [0,0,0,0]

                            # 对装甲测距，切换是存在抖动，采用bbx缩放测距
                            k = 1/5 # bbx尺寸缩放系数
                            k_w = 1/2 # 宽度1/2
                            k_h = 4/5 # 高度4/5（uv坐标）
                            rect[2] = int(  (FinalResult[i].x2 - FinalResult[i].x1)*k  ) # w
                            rect[3] = int(  (FinalResult[i].y2 - FinalResult[i].y1)*k  ) # h
                            rect[0] = int(  FinalResult[i].x1 + (FinalResult[i].x2 - FinalResult[i].x1)*k_w - rect[2]/2  ) # minx
                            rect[1] = int(  FinalResult[i].y1 + (FinalResult[i].y2 - FinalResult[i].y1)*k_h - rect[3]/2 ) # miny

                            # 原有测距方法
                            # if ((FinalResult[i].color == red) | (FinalResult[i].color == blue)): # 可以区分编号，对装甲测距
                            #     rect[0] = int(FinalResult[i].armor_x1) # minx
                            #     rect[1] = int(FinalResult[i].armor_y1) # miny
                            #     rect[2] = int(FinalResult[i].armor_x2 - FinalResult[i].armor_x1) # w
                            #     rect[3] = int(FinalResult[i].armor_y2 - FinalResult[i].armor_y1) # h
                            # else: # 无法区分编号，对bbx w=1/2，h=2/5处作为中心，bbx缩放k倍测距
                            #     k = 1/5 # bbx尺寸缩放系数
                            #     k_w = 1/2 # 宽度1/2
                            #     k_h = 4/5 # 高度4/5（uv坐标）
                            #     rect[2] = int(  (FinalResult[i].x2 - FinalResult[i].x1)*k  ) # w
                            #     rect[3] = int(  (FinalResult[i].y2 - FinalResult[i].y1)*k  ) # h
                            #     rect[0] = int(  FinalResult[i].x1 + (FinalResult[i].x2 - FinalResult[i].x1)*k_w - rect[2]/2  ) # minx
                            #     rect[1] = int(  FinalResult[i].y1 + (FinalResult[i].y2 - FinalResult[i].y1)*k_h - rect[3]/2 ) # miny

                            lidar_coor = self._radar[camera_id].detect_depth([rect]).reshape(-1) # 4.27 radar类内部已经封装好了外参
                            # x = lidar_coor[0] # 图像归一化坐标系坐标
                            # y = lidar_coor[1]
                            # center_x = frame0.shape[1]/2 + frame0.shape[1]/2 * x
                            # center_y = frame0.shape[0]/2 + frame0.shape[0]/2 * y
                            # cv2.circle(frame0, (int(center_x), int(center_y)), 4, (0, 0, 255), 4)
                            
                            # lidar_coor_world = (  T_c2w @ np.concatenate( [np.concatenate([lidar_coor[:2], np.ones(1)], axis=0) * lidar_coor[2], np.ones(1)], axis=0 )  )[:3] # sjtu code
                            lidar_coor_world = np.concatenate([lidar_coor[:2], np.ones(1)], axis=0) # 归一化相机坐标系xy z
                            lidar_coor[2] = xyCoorCompensator(lidar_coor[2]) # 距离补偿
                            lidar_coor_world = np.concatenate( [lidar_coor_world * lidar_coor[2], np.ones(1)], axis=0 ) # 雷达坐标系xyz
                            if camera_id == 0:
                                lidar_coor_world = T_c2w_0 @ lidar_coor_world # 世界坐标系
                            elif camera_id == 1:
                                lidar_coor_world = T_c2w_1 @ lidar_coor_world # 世界坐标系
                            elif camera_id == 2:
                                lidar_coor_world = T_c2w_2 @ lidar_coor_world # 世界坐标系
                                
                            lidar_coor_world = lidar_coor_world[:3]

                            # 红蓝方坐标转换（红方停机坪为原点）28*15场地，红蓝反时，目标坐标进行中心对称变换，蓝方停机坪为原点
                            if reverse:
                                if home_test:
                                    if alliance_game: # 高校联盟5*5场地
                                        lidar_coor_world[0] = 5 - lidar_coor_world[0] # X
                                        lidar_coor_world[1] = -(5 + lidar_coor_world[1]) # Y
                                    else:
                                        lidar_coor_world[0] = ARER_LENGTH - lidar_coor_world[0] # X
                                        lidar_coor_world[1] = -(AREA_WIDTH + lidar_coor_world[1]) # Y
                                else:
                                    lidar_coor_world[0] = ARER_LENGTH - lidar_coor_world[0] # X
                                    lidar_coor_world[1] = -(AREA_WIDTH + lidar_coor_world[1]) # Y

                            # 坐标边界限制
                            if (lidar_coor_world[0] != lidar_coor_world[0]) | (lidar_coor_world[1] != lidar_coor_world[1]) | (lidar_coor_world[2] != lidar_coor_world[2]): # 滤除雷达中的nan
                                lidar_coor_world = [0, 0, 0]
                            if (lidar_coor_world[0] < 0) | (lidar_coor_world[0] > ARER_LENGTH): # X limit
                                lidar_coor_world = [0, 0, 0]
                            if (lidar_coor_world[1] < -AREA_WIDTH) | (lidar_coor_world[1] > 0): # Y limit
                                lidar_coor_world = [0, 0, 0]
                            if (lidar_coor_world[2] < -1) | (lidar_coor_world[2] > 5): # Z limit
                                lidar_coor_world = [0, 0, 0]
                            
                            # 可视化目标世界坐标系下坐标(debug)
                            # 5.3 对应不同相机在画面绘制
                            # if debug_info:
                            x_demo_coor = int(FinalResult[i].x1)
                            y_demo_coor = int(FinalResult[i].y2) -10
                            xyz_string = 'x:' + str(round(lidar_coor_world[0],1)) + ' y:' + str(round(lidar_coor_world[1],1)) + ' z:' + str(round(lidar_coor_world[2],1))
                            if demo:
                                if camera_id == 0:
                                    cv2.rectangle(result_img_0, (x_demo_coor, y_demo_coor), (x_demo_coor+650, y_demo_coor-50), bbx_color, -1)
                                    cv2.putText(result_img_0, xyz_string, (x_demo_coor,y_demo_coor), cv2.FONT_HERSHEY_COMPLEX, 2, txt_color, 2)
                                elif camera_id == 1:
                                    cv2.rectangle(result_img_1, (x_demo_coor, y_demo_coor), (x_demo_coor+650, y_demo_coor-50), bbx_color, -1)
                                    cv2.putText(result_img_1, xyz_string, (x_demo_coor,y_demo_coor), cv2.FONT_HERSHEY_COMPLEX, 2, txt_color, 2)
                                # elif camera_id == 2:
                                #     cv2.rectangle(result_img_2, (x_demo_coor, y_demo_coor), (x_demo_coor+650, y_demo_coor-50), bbx_color, -1)
                                #     cv2.putText(result_img_2, xyz_string, (x_demo_coor,y_demo_coor), cv2.FONT_HERSHEY_COMPLEX, 2, txt_color, 2)
                                # print('type:{0}-x:{1:.2f}-y:{2:.2f}-z:{3:.2f}' .format('car', lidar_coor_world[0], lidar_coor_world[1], lidar_coor_world[2]))

                            # 小地图车辆位置可视化
                            map_img_x, map_img_y = world_coord_2_map_img_coord(lidar_coor_world, map_img.shape[1], map_img.shape[0]) # 世界坐标x,y转换为像素坐标x,y
                            draw_car_circle(map_img, (map_img_x, map_img_y), FinalResult[i].color, FinalResult[i].num) # UI中车辆圆圈绘制

                            # 裁判系统通信相关
                            # 填入敌方信息
                            alarm_color = red if reverse else blue
                            if FinalResult[i].color == alarm_color: # 敌方车辆
                                # 填入敌方单位坐标、预警区域信息
                                num = FinalResult[i].num
                                if (num<0) | (num>6):
                                    logger.warning('越界，序号: {}' .format(num))
                                    num = 0
                                Car_Info[num].id = FinalResult[i].num
                                Car_Info[num].x = lidar_coor_world[0]
                                Car_Info[num].y = lidar_coor_world[1]
                                Car_Info[num].z = lidar_coor_world[2]
                                Car_Info[num].alarm = False
                                Car_Info[num].area = 0
                                if coor_alarm:# 世界坐标系下 遍历预警区域 判断敌方车辆是否进入凸四边形预警区域(需要考虑遮挡造成的测距不准)，坐标与图像预警2选1
                                    # 预警（依赖于雷达测距后世界坐标）
                                    # 4.28 感觉预警的话右边相机不需要用到，所以沿用原来的预警区域
                                    for key,value in alarm_region.items(): # 遍历预警区域
                                        xy_point_arr = np.array([alarm_region[key][0][0], alarm_region[key][0][1], alarm_region[key][1][0], alarm_region[key][1][1], alarm_region[key][2][0], alarm_region[key][2][1], alarm_region[key][3][0], alarm_region[key][3][1]]).reshape(4, 2)
                                        xy_point_arr = abs(xy_point_arr)
                                        box = np.array([tuple(xy_point_arr[0]), tuple(xy_point_arr[1]), tuple(xy_point_arr[2]), tuple(xy_point_arr[3])]) # 预警区域个角点世界坐标xy
                                        center_x = abs(lidar_coor_world[0])
                                        center_y = abs(lidar_coor_world[1])
                                        point = np.array([center_x,center_y]) # 待判断目标中心世界坐标
                                        if is_inside(box.reshape(4, 2), point):
                                            logger.info("xy {0} inside {1}" .format(id_string, key))
                                            Car_Info[num].alarm = True
                                            Car_Info[num].area = key
                                if uv_alarm: # 像素坐标系下 遍历预警区域 判断敌方车辆是否进入凸四边形预警区域(需要考虑透视遮挡关系)
                                    for key,value in alarm_region.items(): # 遍历预警区域
                                        uv_point_arr = point_3D_to_2D_coor_4points(alarm_region[key][0], alarm_region[key][1], alarm_region[key][2], alarm_region[key][3], T_w2c_1, self._K_0)
                                        box = np.array([tuple(uv_point_arr[0]), tuple(uv_point_arr[1]), tuple(uv_point_arr[2]), tuple(uv_point_arr[3])]) # 预警区域个角点像素坐标uv
                                        center_x = int((FinalResult[i].armor_x1 + FinalResult[i].armor_x2)/2)
                                        center_y = int((FinalResult[i].armor_y1 + FinalResult[i].armor_y2)/2)
                                        point = np.array([center_x,center_y]) # 待判断目标中心像素坐标
                                        if is_inside(box.reshape(4, 2), point):
                                            logger.info("uv {0} inside {1}" .format(id_string, key))
                                            Car_Info[num].alarm = True
                                            Car_Info[num].area = key
                                    
                            # 填入己方信息
                            our_color = blue if reverse else red
                            if FinalResult[i].color == our_color: # 我方车辆
                                num = FinalResult[i].num
                                if (num<0) | (num>6):
                                    logger.warning('越界，序号: {}' .format(num))
                                    num = 0
                                Our_Car_Info[num].id = FinalResult[i].num
                                Our_Car_Info[num].x = lidar_coor_world[0]
                                Our_Car_Info[num].y = lidar_coor_world[1]
                                Our_Car_Info[num].z = lidar_coor_world[2]
            
            
            t9 = time.time()
            print("后处理时间: ".format(t9-t3))
            
            # 05 预警与坐标发送
            # 敌方坐标信息填入
            # 联盟赛，只填入5号坐标
            if alliance_game:
                enemy_location = {}
                for i in range(1, 7):
                    enemy_location[str(i)] = [0, 0]
                for i in range(5, 0, -1):
                    x = Car_Info[i].x
                    y = Car_Info[i].y
                    if x!=0 and y!=0: # 只找第一个均不为0的
                        enemy_location[str(5)] = [x, y]
                        break
                enemy_coor_message = {'task': 1, 'data': [enemy_location]}
                send_judge(enemy_coor_message, QTwindow) # 向串口队列中填入坐标信息
                logger.debug("enemy_coor_message = {}" .format(enemy_coor_message))
            else:
                enemy_location = {}
                for i in range(1, 7):
                    x = Car_Info[i].x
                    y = Car_Info[i].y
                    enemy_location[str(i)] = [x, y]
                enemy_coor_message = {'task': 1, 'data': [enemy_location]}
                send_judge(enemy_coor_message, QTwindow) # 向串口队列中填入坐标信息
                logger.debug("enemy_coor_message = {}" .format(enemy_coor_message))

            # 我方坐标信息填入
            # 联盟赛，只填入5号坐标
            if alliance_game:
                our_location = {}
                for i in range(1, 7):
                    our_location[str(i)] = [0, 0]
                for i in range(5, 0, -1):
                    x = Our_Car_Info[i].x
                    y = Our_Car_Info[i].y
                    if x!=0 and y!=0: # 只找第一个均不为0的
                        our_location[str(5)] = [x, y]
                        break
                our_coor_message = {'task': 3, 'data': [our_location]}
                send_judge(our_coor_message, QTwindow) # 向串口队列中填入坐标信息
                logger.debug("our_coor_message = {}" .format(our_coor_message))
            else :
                our_location = {}
                for i in range(1, 7):
                    x = Our_Car_Info[i].x
                    y = Our_Car_Info[i].y
                    our_location[str(i)] = [x, y]
                our_coor_message = {'task': 3, 'data': [our_location]}
                send_judge(our_coor_message, QTwindow) # 向串口队列中填入坐标信息
                logger.debug("our_coor_message = {}" .format(our_coor_message))

            # 预警信息填入
            for loc,value in alarm_region.items(): # 遍历预警区域，对每个区域内进入的地方单位进行预警
                alarm_target = [] # 车辆编号
                team = 'red' if reverse else 'blue'
                for i in range(1,7):
                    if (Car_Info[i].area==loc):
                        alarm_target.append(i)
                if len(alarm_target): # 预警区域内进入车辆
                    alarm_message = {'task': 2, 'data': [loc, alarm_target, team]}
                    send_judge(alarm_message, QTwindow) # 向串口队列中填入预警信息
                    logger.debug("alarm_message = {}" .format(alarm_message["data"]))

            # cv2.circle(result_img, (int(frame0.shape[1]/2), int(frame0.shape[0]/2)), 10, (0, 255, 0), -1) # 激光归零，图像中心点绘制

            if debug_info:
                logger.debug("---------------")
            

            # 帧率计算
            fps = 1 / (time.time() - t_spin_once)
            # if not self._fps_queue.full():
            #     self._fps_queue.put(fps)
            # else:
            #     self._fps_queue.get()
            # sum = 0
            # if not self._fps_queue.empty():
            #     for i in range(1, self._fps_queue.qsize()):
            #         tmp = self._fps_queue.get()
            #         self._fps_queue.put(tmp)
            #         sum += tmp
            #     fps = sum/self._fps_queue.qsize()
            logger.info("frame: {0} FPS: {1:.3f}" .format(self._frame_num, fps))
            print(car_number)
            fps_string = 'FPS: ' + str(int(fps))

            # 三个相机视频源绘制fps
            if QTwindow.camera0_view:
                cv2.rectangle(result_img_0, (0, 0), (300, 80), (0, 0, 0), -1)
                cv2.putText(result_img_0, fps_string, (20,60), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
            elif QTwindow.camera1_view:
                cv2.rectangle(result_img_1, (0, 0), (300, 80), (0, 0, 0), -1)
                cv2.putText(result_img_1, fps_string, (20,60), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
            # elif QTwindow.camera2_view:
            #     cv2.rectangle(result_img_2, (0, 0), (300, 80), (0, 0, 0), -1)
            #     cv2.putText(result_img_2, fps_string, (20,60), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)

            if using_video:
                pass
                # time.sleep(0.08)
            
            # 4.28 实际上就是从frame0/frame1里截取
            # 5.5 现在测试比较难受，因为外参都默认是中间相机的，导致截取roi可能会出界(超出图片长宽)，建议之后旁边两个相机roi区域世界坐标改一下，测试的时候就先用一个imgsz好了
            # 千万注意名字不要冲突了
            # 5.13 两侧相机暂时不知道用一个什么画面中心来截取，可能会导致异常错误，先注释
            if roi:
                if self._position_flag and QTwindow.camera0_view:
                    draw_3d_point_in_uv(CAM_CENTER_L, T_w2c_0, self._K_0[0], result_img_0, 2)
                    draw_3d_point_in_uv(CAM_CENTER_R, T_w2c_0, self._K_0[0], result_img_0, 2)
                    draw_3d_point_in_uv(CAM_CENTER_BASE, T_w2c_0, self._K_0[0], result_img_0, 2)
                    # frame3 = get_roi_img(CAM_CENTER_L, T_w2c_0, self._K_0[0], frame0, self._imgsz[0], ROI_SIZE)
                    # frame4 = get_roi_img(CAM_CENTER_R, T_w2c_0, self._K_0[0], frame0, self._imgsz[0], ROI_SIZE)
                    # frame_base = get_roi_img(CAM_CENTER_BASE, T_w2c_0, self._K_0[0], frame0, self._imgsz[0], ROI_BASE_SIZE)
                elif self._position_flag and QTwindow.camera1_view:
                    pass
                    draw_3d_point_in_uv(CAM_CENTER_L, T_w2c_1, self._K_0[1], result_img_1, 2)
                    draw_3d_point_in_uv(CAM_CENTER_R, T_w2c_1, self._K_0[1], result_img_1, 2)
                    draw_3d_point_in_uv(CAM_CENTER_BASE, T_w2c_1, self._K_0[1], result_img_1, 2)
                    # frame3 = get_roi_img(CAM_CENTER_L, T_w2c_1, self._K_0[1], frame1, self._imgsz[1], ROI_SIZE)
                    # frame4 = get_roi_img(CAM_CENTER_R, T_w2c_1, self._K_0[1], frame1, self._imgsz[1], ROI_SIZE)
                    # frame_base = get_roi_img(CAM_CENTER_BASE, T_w2c_1, self._K_0[1], frame1, self._imgsz[1], ROI_BASE_SIZE)
                # elif self._position_flag and QTwindow.camera2_view:
                #     pass
                #     draw_3d_point_in_uv(CAM_CENTER_L, T_w2c_2, self._K_0[2], result_img_2, 2)
                #     draw_3d_point_in_uv(CAM_CENTER_R, T_w2c_2, self._K_0[2], result_img_2, 2)
                #     draw_3d_point_in_uv(CAM_CENTER_BASE, T_w2c_2, self._K_0[2], result_img_2, 2)
                    # frame3 = get_roi_img(CAM_CENTER_L, T_w2c_2, self._K_0[2], frame2, self._imgsz[2], ROI_SIZE)
                    # frame4 = get_roi_img(CAM_CENTER_R, T_w2c_2, self._K_0[2], frame2, self._imgsz[2], ROI_SIZE)
                    # frame_base = get_roi_img(CAM_CENTER_BASE, T_w2c_2, self._K_0[2], frame2, self._imgsz[2], ROI_BASE_SIZE)

            # opencv window
            # 5.5 可以用来检查雷达连上没有
            if not using_dq:
                # t0 = time.time()
                cv2.imshow("depth", depth) # 5ms
                # print((time.time() - t0) * 1000)
            else: # 使用点云队列
                # cv2.imshow("result_img", result_img) # 2.9ms
                pass

            # 4.28 这部分主要是调试用，实际上也调试不出什么
            if debug_info:
                if QTwindow.camera0_view:
                    pass
                    # cv2.imshow("frame0", frame0)
                    # cv2.imshow('yolo_result_img_0', yolo_result_img_0)
                    # if roi and self._position_flag:
                    #     cv2.imshow("ROI_L", frame3)
                    #     cv2.imshow("ROI_R", frame4)
                    #     cv2.imshow("ROI_BASE", frame_base)
                elif QTwindow.camera1_view:
                    pass
                    # cv2.imshow("frame1", frame1)
                    # cv2.imshow('yolo_result_img_1', yolo_result_img_1)
                    # if roi and self._position_flag:
                    #     cv2.imshow("ROI_L", frame3)
                    #     cv2.imshow("ROI_R", frame4)
                    #     cv2.imshow("ROI_BASE", frame_base)
                # elif QTwindow.camera2_view:
                #     pass
                    # cv2.imshow("frame1", frame2)
                    # cv2.imshow('yolo_result_img_2', yolo_result_img_2)
                    # if roi and self._position_flag:
                    #     cv2.imshow("ROI_L", frame3)
                    #     cv2.imshow("ROI_R", frame4)
                    #     cv2.imshow("ROI_BASE", frame_base)
            
        # UI窗口可视化
        # 4.28 这部分应该就是切换和呈现相机视频源，btn3负责修改camera0_view
        if flag0 and QTwindow.camera0_view:
            result_img_0 = cv2.resize(result_img_0, (int(result_img_0.shape[1]/2), int(result_img_0.shape[0]/2)))
            QTwindow.set_image(result_img_0, "main_demo") 
        
        elif flag1 and QTwindow.camera1_view:
            result_img_1 = cv2.resize(result_img_1, (int(result_img_1.shape[1]/2), int(result_img_1.shape[0]/2)))
            QTwindow.set_image(result_img_1, "main_demo") 
        # elif flag2 and QTwindow.camera2_view:
        #     result_img_2 = cv2.resize(result_img_2, (int(result_img_2.shape[1]/2), int(result_img_2.shape[0]/2)))
        #     QTwindow.set_image(result_img_2, "main_demo") 
        
        QTwindow.set_image(map_img, "map")  # 小地图
        
        # 4.28 似乎需要opencv先imshow()之后再这边set_image
        # 5.13 前面注释之后，分开的另外两个相机切换视角的时候roi呈现静态的图
        # if roi and self._position_flag and (flag0 or flag1 or flag2):
        #     try:
        #         QTwindow.set_image(frame3, "second_cam")
        #         QTwindow.set_image(frame4, "third_cam")
        #         QTwindow.set_image(frame_base, "base_img")
        #     except:
        #         logger.error("roi err")

if __name__ == '__main__':

    # log
    logger = logging.getLogger('logger')
    if debug_info:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    formator = logging.Formatter(fmt="%(asctime)s [ %(filename)s ]  %(lineno)d行 | [ %(levelname)s ] | [%(message)s]", datefmt="%Y/%m/%d/%X")
    sh = logging.StreamHandler()  # 创建一个输出的处理器，让它输入到控制台
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    log_name_appendex = "{0}.txt".format(datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
    filename = os.path.join(LOG_DIR, log_name_appendex)
    fh = logging.FileHandler(filename, encoding="utf-8")  # 创建一个把日志信息存储到文件中的处理器
    logger.addHandler(sh)  # 把输出处理器添加到日志器中
    sh.setFormatter(formator)
    logger.addHandler(fh)  # 把文件处理器，加载到logger中
    fh.setFormatter(formator)  # 给文件处理器添加格式器
    logger.debug('debug_log_test')
    logger.info('info_log_test')
    logger.warning('warning_log_test')
    logger.error('error_log_test')
    logger.critical('critical_log_test')

    ###### TODO:串口通信设置，你如果暂时不用串口通信，可以将它们注释掉，且该模块只能在linux下运行########
    # sudo chmod 777 /dev/ttyUSB0
    if using_serial:
        logger.info("try to open Serial")
        password = PASS_WORD
        ch = pexpect.spawn('sudo chmod 777 {}'.format(USB))
        ch.sendline(password)
        logger.info("set password ok")

        if debug_info: # 调试情况下打开ttyUSB1
            ch = pexpect.spawn('sudo chmod 777 /dev/ttyUSB1')
            ch.sendline(password)
            
        ser = serial.Serial(USB, 115200, timeout=0.2)
        if ser.is_open:
            logger.info("Serial is open")
            ser.flushInput()
        else:
            ser.open()
        
        # 串口读写线程开启
        _thread.start_new_thread(read, (ser,))
        _thread.start_new_thread(write, (ser,))
    ##########################################################

    # UI
    app = QtWidgets.QApplication(sys.argv)
    QTwindow = Mywindow()
    QTwindow.show()
    SELF_COLOR = ['RED', 'BLUE']
    QTwindow.set_text("message_box", f"You are {SELF_COLOR[reverse]:s}") # 初始化显示，enemy是否选对

    # 这个应该是UI 里面的血条初始化，主函数内其他地方就没有了
    hp_scene = HP_scene(reverse, lambda x: QTwindow.set_image(x, "message_box"))
    
    
    t4 = time.time()
    
    try:
        main_process = radar_process(QTwindow)
        t5 = time.time()
        print("init time :{0}".format(t5-t4))
        
        key_close = False # 按键终止程序

        while True:
            close_flag = False # 终止条件

            # 显示hp信息
            # 5.7 get_message注释掉了，其实调试的时候不开serial都是保持有东西的，应该没啥影响
            UART_passer.get_message(hp_scene)
            hp_scene.show()  # update the message box
            
            # 记录标记进度
            logger.info("mark hero progress is {0}".format(UART_passer.mark_hero))
            logger.info("mark engineer progress is {0}".format(UART_passer.mark_engineer))
            logger.info("mark standard 3 progress is {0}".format(UART_passer.mark_standard_3))
            logger.info("mark standard 4 progress is {0}".format(UART_passer.mark_standard_4))
            logger.info("mark standard 5 progress is {0}".format(UART_passer.mark_standard_5))
            logger.info("mark sentry progress is {0}".format(UART_passer.mark_sentry))
            
            # 主循环
            main_process.spin_once()

            k = cv2.waitKey(1)
        
            # 这些都是给雷达站准备人员，云台手操作不了
            if k == 0xff & ord("q"):
                close_flag = True
                logger.fatal("USER CLOSED")
            elif k == 0xff & ord("p"): # 暂停
                logger.info("USER PAUSED")
                cv2.waitKey(0)

            if close_flag:
                break
    except Exception as e:
        traceback.print_exc()
        QTwindow.set_text("feedback", "[ERROR] Program broken")
        pass

    # 若在录制过程结束录制
    # 5.3 这里要不要改？ 一般是进不到这里，估计这是中途退出帮助结束录制
    if QTwindow.record_state[0] or QTwindow.record_state[1] or QTwindow.record_state[2]:
        QTwindow.btn2_clicked()

    if not using_dq:
        Radar.stop()

    cv2.destroyAllWindows()
    QTwindow.close()

    logger.fatal("Main Program Finished")

    sys.exit(app.exec_())
