'''
common.py
对于所有类都有用的函数
'''
import numpy as np
import cv2
import time
import os
from datetime import datetime

from radar_module.config import red, blue, grey, reverse, ARER_LENGTH, AREA_WIDTH, home_test, alliance_game, VIDEO_SAVE_DIR
from radar_module.camera import read_yaml 
import ui

def is_inside(box: np.ndarray, point: np.ndarray):
    '''
    判断点是否在凸四边形中

    :param box:为凸四边形的四点 shape is (4,2)
    :param point:为需判断的是否在内的点 shape is (2,)
    '''

    # 使用案例
    # box = np.array([[0,0],[3,0],[3,3],[0,3]])
    # point = np.array([1,1])
    # if is_inside(box.reshape(4, 2),point):
    #     print('inside')

    assert box.shape == (4, 2)
    assert point.shape == (2,)
    AM = point - box[0]
    AB = box[1] - box[0]
    BM = point - box[1]
    BC = box[2] - box[1]
    CM = point - box[2]
    CD = box[3] - box[2]
    DM = point - box[3]
    DA = box[0] - box[3]
    a = np.cross(AM, AB)
    b = np.cross(BM, BC)
    c = np.cross(CM, CD)
    d = np.cross(DM, DA)
    return a >= 0 and b >= 0 and c >= 0 and d >= 0 or \
           a <= 0 and b <= 0 and c <= 0 and d <= 0

def point_3D_to_2D_coor_4points(point_3D_L_T, point_3D_L_B, point_3D_R_B, point_3D_R_T, T_w2c, K_0):
    '''
    将世界坐标系下四个三维坐标点，转换为像素坐标系下坐标

    :param point_3D_L_T:point_3D_L_T
    :param point_3D_L_B:point_3D_L_B
    :param point_3D_R_B:point_3D_R_B
    :param point_3D_R_T:point_3D_R_T
    :param T_w2c:世界到相机变换矩阵
    :param K_0:相机内参矩阵K_0

    :return: uv_point_arr 像素坐标系下坐标x、y数组
    '''
    point_2D_L_T_x, point_2D_L_T_y = point_3D_to_2D(point_3D_L_T, T_w2c, K_0)
    point_2D_L_B_x, point_2D_L_B_y = point_3D_to_2D(point_3D_L_B, T_w2c, K_0)
    point_2D_R_B_x, point_2D_R_B_y = point_3D_to_2D(point_3D_R_B, T_w2c, K_0)
    point_2D_R_T_x, point_2D_R_T_y = point_3D_to_2D(point_3D_R_T, T_w2c, K_0)

    uv_point_arr = np.array([point_2D_L_T_x, point_2D_L_T_y, point_2D_L_B_x, point_2D_L_B_y, point_2D_R_B_x, point_2D_R_B_y, point_2D_R_T_x, point_2D_R_T_y]).reshape(4, 2)
    return uv_point_arr

def point_3D_to_2D(point_3D, T_w2c, K_0):
    '''
    将世界坐标系下三维坐标多点投射并绘制在像素坐标系下

    :param point_3D:三维坐标
    :param T_w2c:世界到相机变换矩阵
    :param K_0:相机内参矩阵K_0

    :return: point_2D_x,point_2D_y 像素坐标系下坐标x、y
    '''
    point_cam = (T_w2c@point_3D)[:3] # 转换到相机坐标系
    # print('point_cam[-1] = {}' .format(point_cam[-1]))
    # if point_cam[-1] == 0:
    #     point_cam[-1] = 1e-5
    # print('point_cam[-1] = {}' .format(point_cam[-1]))
    point_2D = (K_0 @ point_cam)/point_cam[-1] # 转换到像素坐标系
    
    point_2D_x = int(point_2D[0])
    point_2D_y = int(point_2D[1])
    # print(point_2D_x)
    # print(point_2D_y)
    return point_2D_x,point_2D_y

def draw_alarm_area(point_3D_L_T, point_3D_L_B, point_3D_R_B, point_3D_R_T, T_w2c, K_0, img, thickness):
    '''
    将世界坐标系下三维坐标多点投射并绘制在像素坐标系下

    :param point_3D_L_T:point_3D_L_T
    :param point_3D_L_B:point_3D_L_B
    :param point_3D_R_B:point_3D_R_B
    :param point_3D_R_T:point_3D_R_T
    :param T_w2c:世界到相机变换矩阵
    :param K_0:相机内参矩阵K_0
    :param frame:图片
    '''
    point_2D_L_T_x, point_2D_L_T_y = point_3D_to_2D(point_3D_L_T, T_w2c, K_0)
    # cv2.circle(img, (point_2D_L_T_x,point_2D_L_T_y), 3, (0,255,0), 3, 0)
    point_2D_L_B_x, point_2D_L_B_y = point_3D_to_2D(point_3D_L_B, T_w2c, K_0)
    # cv2.circle(img, (point_2D_L_B_x,point_2D_L_B_y), 3, (0,255,0), 3, 0)
    point_2D_R_B_x, point_2D_R_B_y = point_3D_to_2D(point_3D_R_B, T_w2c, K_0)
    # cv2.circle(img, (point_2D_R_B_x,point_2D_R_B_y), 3, (0,255,0), 3, 0)
    point_2D_R_T_x, point_2D_R_T_y = point_3D_to_2D(point_3D_R_T, T_w2c, K_0)
    # cv2.circle(img, (point_2D_R_T_x,point_2D_R_T_y), 3, (0,255,0), 3, 0)
    
    uv_point_arr = np.array([point_2D_L_T_x, point_2D_L_T_y, point_2D_L_B_x, point_2D_L_B_y, point_2D_R_B_x, point_2D_R_B_y, point_2D_R_T_x, point_2D_R_T_y]).reshape(4, 2)

    for i in range(4):
        cv2.line(img, tuple(uv_point_arr[i]), tuple(uv_point_arr[(i + 1) % 4]), (0, 255, 0), thickness)

def draw_car_circle(img, location, color, armor_num: int):
    '''
    画车辆定位点

    :param img:输入图像
    :param location:坐标[x,y](世界坐标)
    :param color:预测出的车辆颜色
    :param armor_num:预测出的车辆编号
    '''
    if color == red:
        draw_color = (0, 0, 255)
    elif color == blue:
        draw_color = (255, 0, 0)
    else:
        draw_color = (0, 0, 0)
    circle_size = 15
    cv2.circle(img, tuple(location), circle_size, draw_color, -1)  # 内部填充
    cv2.circle(img, tuple(location), circle_size, (0, 0, 0), 1)  # 外边框
    cv2.putText(img, str(armor_num),
                (location[0] - 7 * circle_size // 10, location[1] + 6 * circle_size // 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * circle_size / 10, (255, 255, 255), 2)# 数字

def world_coord_2_map_img_coord(world_coord, w, h):
    '''
    将赛场世界坐标转换为平面图像坐标

    :param world_coord:赛场世界坐标
    :param map_img_coord:平面图像坐标
    :param w:img_w
    :param h:img_h
    '''
    map_img_coord = [0,0]
    if home_test:
        if alliance_game: # 联盟赛场地5*5
            map_img_coord[0] = int((-world_coord[1]) / 5 * w)
            map_img_coord[1] = int((5 - world_coord[0]) / 5 * h)
        else:
            map_img_coord[0] = int((-world_coord[1]) / AREA_WIDTH * w)
            map_img_coord[1] = int((ARER_LENGTH - world_coord[0]) / ARER_LENGTH * h)
    else:
        map_img_coord[0] = int((-world_coord[1]) / AREA_WIDTH * w)
        map_img_coord[1] = int((ARER_LENGTH - world_coord[0]) / ARER_LENGTH * h)

    return map_img_coord

def draw_alarm_area_map(point_3D_L_T, point_3D_L_B, point_3D_R_B, point_3D_R_T, img, thickness):
    '''
    将世界坐标系下三维坐标点投射并绘制在小地图中

    :param point_3D_L_T:point_3D_L_T
    :param point_3D_L_B:point_3D_L_B
    :param point_3D_R_B:point_3D_R_B
    :param point_3D_R_T:point_3D_R_T
    :param frame:图片
    '''
    h,w = img.shape[0:2]
    # print('w,h = {0},{1}'.format(w,h))
    LT_x, LT_y = world_coord_2_map_img_coord((point_3D_L_T[0], point_3D_L_T[1]), w, h)
    LB_x, LB_y = world_coord_2_map_img_coord((point_3D_L_B[0], point_3D_L_B[1]), w, h)
    RB_x, RB_y = world_coord_2_map_img_coord((point_3D_R_B[0], point_3D_R_B[1]), w, h)
    RT_x, RT_y = world_coord_2_map_img_coord((point_3D_R_T[0], point_3D_R_T[1]), w, h)

    if reverse: # 预警区域坐标为我方视角坐标，敌我反向时，绘制坐标需要反向
        LT_x = w-LT_x
        LB_x = w-LB_x
        RB_x = w-RB_x
        RT_x = w-RT_x
        LT_y = h-LT_y
        LB_y = h-LB_y
        RB_y = h-RB_y
        RT_y = h-RT_y
    
    map_point_arr = np.array([LT_x, LT_y, LB_x, LB_y, RB_x, RB_y, RT_x, RT_y]).reshape(4, 2)

    for i in range(4):
        cv2.line(img, tuple(map_point_arr[i]), tuple(map_point_arr[(i + 1) % 4]), (0, 255, 0), thickness)
    

def xyCoorCompensator(lidarCoorX):
    # 将测距点距离补偿到车辆中心坐标(lidar坐标系下x轴距离)
    return lidarCoorX-0.5

def draw_3d_point_in_uv(point_3d, T_w2c, K_0, img, thickness):
    x, y = point_3D_to_2D(point_3d, T_w2c, K_0)
    cv2.circle(img, (x,y), 5, (0,255,0), thickness)

def get_roi_img(point_3d_center, T_w2c, K_0, img, img_size, roi_size):
    x, y = point_3D_to_2D(point_3d_center, T_w2c, K_0)
    x0 = int(x-roi_size[0]/2) if(int(x-roi_size[0]/2) >= 0) else 0
    y0 = int(y-roi_size[1]/2) if(int(y-roi_size[1]/2) >= 0) else 0
    x1 = int(x+roi_size[0]/2) if(int(x+roi_size[0]/2) <= img_size[0]) else img_size[0]
    y1 = int(y+roi_size[1]/2) if(int(y+roi_size[1]/2) <= img_size[1]) else img_size[1]
    img = img[y0:y1, x0:x1] # 裁剪ROI
    if(img.shape[1] != roi_size[0] or img.shape[0] != roi_size[1]):
        img = cv2.resize(img, (roi_size[0],roi_size[1])) # 长宽比不一致PYQT显示有问题
    return img


# 6.1 单相机多线程录像效果良好，迁移到多相机
# 为了充分发挥多线程的作用，每个相机录像用一个线程
def middle_record(cap):
    # 用全局变量的话可能主程序来不及修改
    save_address = VIDEO_SAVE_DIR
    # 找不到录制文件夹就终止线程
    if not os.path.exists(save_address):
        print("[ERROR] path not existing")
        return
    title = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    os.mkdir(os.path.join(save_address,title))
    name = os.path.join(save_address, title, "camera0.mp4")
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    camera_0_size = read_yaml(0)[4]
    record_fps = 25
    record_object = cv2.VideoWriter(name, fourcc, record_fps, camera_0_size)
      
    
    while True:
        if ui.stop_record:
            record_object.release()
            break
        cap._lock.acquire()
        flag, frame = cap.read()
        cap._lock.release()
        if flag:
            record_object.write(frame)
        time.sleep(0.04)  # 控制录像帧率
        

def left_record(cap):
    # 用全局变量的话可能主程序来不及修改
    save_address = VIDEO_SAVE_DIR
    # 找不到录制文件夹就终止线程
    if not os.path.exists(save_address):
        print("[ERROR] path not existing")
        return
    title = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    # os.mkdir(os.path.join(save_address,title))
    name = os.path.join(save_address, title, "camera1.mp4")
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    camera_1_size = read_yaml(1)[4]
    record_fps = 25
    record_object = cv2.VideoWriter(name, fourcc, record_fps, camera_1_size)
      
    
    while True:
        if ui.stop_record:
            record_object.release()
            break
        cap._lock.acquire()
        flag, frame = cap.read()
        cap._lock.release()
        if flag:
            record_object.write(frame)
        time.sleep(0.04)  # 控制录像帧率
        
        
# def right_record(cap):
#     # 用全局变量的话可能主程序来不及修改
#     save_address = VIDEO_SAVE_DIR
#     # 找不到录制文件夹就终止线程
#     if not os.path.exists(save_address):
#         print("[ERROR] path not existing")
#         return
#     title = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
#     # os.mkdir(os.path.join(save_address,title))
#     # 6.1 确保输入的视频尺寸和相机尺寸一致，否则保存的视频无法播放
#     name = os.path.join(save_address, title, "camera1.mp4")
#     fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#     camera_1_size = read_yaml(1)[4]
#     record_fps = 25
#     record_object = cv2.VideoWriter(name, fourcc, record_fps, camera_1_size)
      
    
#     while True:
#         if ui.stop_record:
#             record_object.release()
#             break
#         cap._lock.acquire()
#         flag, frame = cap.read()
#         cap._lock.release()
#         if flag:
#             record_object.write(frame)
#         time.sleep(0.04)  # 控制录像帧率