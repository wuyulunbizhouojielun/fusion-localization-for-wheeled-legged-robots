'''
位姿估计函数
进行手动位姿估计
'''
import cv2
import numpy as np
import os
from datetime import datetime

from radar_module.camera import Camera_Thread, read_yaml, white_balance
from radar_module.config import LOCATION_SAVE_DIR, location_targets, whiteBalance

def __callback_1(event,x,y,flags,param):
    '''
    鼠标回调函数
    鼠标点击点：确认标定点并在图像上显示
    鼠标位置：用来生成放大图
    '''
    # using EPS and MAX_ITER combine
    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                         30, 0.001)
    if event == cv2.EVENT_MOUSEMOVE:
        # 周围200*200像素放大图
        rect = cv2.getWindowImageRect(param["pick_winname"])
        img_cut = np.zeros((200,200,3),np.uint8)
        img_cut[max(-y+100,0):min(param["pick_img"].shape[0]+100-y,200),max(-x+100,0):min(param["pick_img"].shape[1]+100-x,200)] = \
        param["pick_img"][max(y-100,0):min(y+100,param["pick_img"].shape[0]),max(x-100,0):min(x+100,param["pick_img"].shape[1])]
        cv2.circle(img_cut,(100,100),1,(0,255,0),1)
        cv2.imshow(param["zoom_winname"], img_cut)
        cv2.moveWindow(param["zoom_winname"],rect[0]-400,rect[1]+200)
        cv2.resizeWindow(param["zoom_winname"], 400,400)
    if event == cv2.EVENT_LBUTTONDOWN and not param["pick_flag"]:
        param["pick_flag"] = True
        print(f"pick ({x:d},{y:d})")
        # 亚像素精确化
        corner = cv2.cornerSubPix(param["pick_img_raw"],np.float32([x,y]).reshape(1,1,2),(5,5),(-1,-1),stop_criteria).reshape(2)
        param["pick_point"] = [corner[0],corner[1]]
        cv2.circle(param["pick_img"],(x,y),2,(0,255,0),1)

def locate_pick(cap:Camera_Thread, reverse, camera_type, home_size = False, video_test = False):
    '''
    手动四点标定

    :param cap:Camera_Thread object
    :param reverse:change r b
    :param camera_type:camera number
    :param home_size: 选用在家里测试时的尺寸
    :param video_test: 是否用视频测试，以减慢播放速度

    :return: 读取成功标志，旋转向量，平移向量
    '''

    # 4.18 针对双相机进行修改
    # reverse 0 我方R 对方B 此时左相机用目前的四个点，右相机用右边的前哨站，必须保证两个相机内都能拍到R0定位标签

    # 4.19 感觉画面并不用切换回对应相机，因为直接读取一张frame，和主视频源无关

    if home_size:
        location_type = 'home_test'
        red_base = location_targets[location_type]['red_base']
        blue_outpost = location_targets[location_type]['blue_outpost']
        red_outpost = location_targets[location_type]['red_outpost']
        blue_base = location_targets[location_type]['blue_base']
        r_rt = location_targets[location_type]['r_rt']
        r_lt = location_targets[location_type]['r_lt']
        b_rt = location_targets[location_type]['b_rt']
        b_lt = location_targets[location_type]['b_lt']
    else:
        location_type = 'game'
        red_base = location_targets[location_type]['red_base']
        blue_outpost = location_targets[location_type]['blue_outpost']
        red_outpost = location_targets[location_type]['red_outpost']
        blue_base = location_targets[location_type]['blue_base']
        r_rt = location_targets[location_type]['r_rt']
        r_lt = location_targets[location_type]['r_lt']
        b_rt = location_targets[location_type]['b_rt']
        b_lt = location_targets[location_type]['b_lt']
    
    # 4.19 两个相机是分开标的，所以这里不用区分
    K_0, C_0= read_yaml(camera_type)[1:3]
    # 窗口下方提示标定哪个目标reverse+camera num
    tips = \
    {
        '00':['enemy_outpost','enemy_base','our_right_top','our_left_top'],
        '10':['enemy_outpost','enemy_base','our_right_top','our_left_top']
    }
    
    # 设定世界坐标
    if not reverse:
        ops = np.float64([blue_outpost, blue_base, r_rt, r_lt])
    else:
        ops = np.float64([red_outpost, red_base, b_rt, b_lt])
    ops = ops.reshape(4,1,3)

    r, frame = cap.read()
    if not cap.is_open():
        return False,None,None

    if whiteBalance:
        frame = white_balance(frame) # 白平衡

    # 标定目标提示位置
    tip_w = frame.shape[1]//2
    tip_h = frame.shape[0]-200

    # OpenCV窗口参数
    info = {}
    info["pick_img"] = frame
    info["pick_img_raw"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    info["pick_winname"] = "pick_corner"
    info["zoom_winname"] = "zoom_in"
    info["pick_flag"] = False
    info["pick_point"] = None # 回调函数中点击的点

    cv2.namedWindow(info["pick_winname"], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(info["pick_winname"], 1280,780)
    cv2.setWindowProperty(info["pick_winname"],cv2.WND_PROP_TOPMOST,1)
    cv2.moveWindow(info["pick_winname"], 500,300)
    cv2.namedWindow(info["zoom_winname"], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(info["zoom_winname"], 400, 400)
    cv2.setWindowProperty(info["zoom_winname"],cv2.WND_PROP_TOPMOST,1)
    cv2.setMouseCallback("pick_corner", __callback_1, info)

    pick_point = []
    while True:
        # draw tips
        cv2.putText(frame,tips[str(reverse)+str(camera_type)][len(pick_point)],(tip_w,tip_h),
                    cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),2)

        # draw the points having been picked
        for select_p in pick_point:
            cv2.circle(frame, (int(select_p[0]), int(select_p[1])), 1, (0, 255, 0), 2)

        # draw the connecting line following the picking order
        for p_index in range(1, len(pick_point)):
            cv2.line(frame, (int(pick_point[p_index - 1][0]), int(pick_point[p_index - 1][1])),
                     (int(pick_point[p_index][0]), int(pick_point[p_index][1])), (0, 255, 0), 2)

        cv2.imshow(info["pick_winname"], info["pick_img"])

        if info["pick_flag"]: # 当在回调函数中触发点击事件
            pick_point.append(info["pick_point"])
            # draw the points having been picked
            for select_p in pick_point:
                cv2.circle(frame, (int(select_p[0]), int(select_p[1])), 1, (0, 255, 0), 2)

            # draw the connecting line following the picking order
            for p_index in range(1, len(pick_point)):
                cv2.line(frame, (int(pick_point[p_index - 1][0]), int(pick_point[p_index - 1][1])),
                         (int(pick_point[p_index][0]), int(pick_point[p_index][1])), (0, 255, 0), 2)
            # 四点完成，首尾相连
            if len(pick_point) == 4:
                cv2.line(frame, (int(pick_point[3][0]), int(pick_point[3][1])),
                         (int(pick_point[0][0]), int(pick_point[0][1])), (0, 255, 0), 2)

            cv2.imshow(info["pick_winname"], info["pick_img"])
            # 将刚加入的pop出等待确认后再加入
            pick_point.pop()
            key = cv2.waitKey(0)
            if key == ord('c') & 0xFF: # 确认点加入
                pick_point.append(info["pick_point"])

                print(f"You have pick {len(pick_point):d} point.")

            if key == ord('z') & 0xFF: # 将上一次加入的点也删除（这次的也不要）
                if len(pick_point):
                    pick_point.pop()
                print("drop last")

            if key == ord('q') & 0xFF: # 直接退出标定，比如你来不及了
                cv2.destroyWindow(info["pick_winname"])
                cv2.destroyWindow(info["zoom_winname"])
                return False,None,None
            info["pick_flag"] = False
        else:
            # 当未点击时，持续输出视频
            if video_test:
                cv2.waitKey(80)
            else:
                cv2.waitKey(1)
        if len(pick_point) == 4:  # 四点全部选定完成，进行PNP
            break
        r, frame = cap.read()

        if whiteBalance:
            frame = white_balance(frame) # 白平衡

        if not cap.is_open():
            cv2.destroyWindow(info["pick_winname"])
            cv2.destroyWindow(info["zoom_winname"])
            return False,None,None
        info["pick_img"] = frame

    pick_point = np.float64(pick_point).reshape(-1,1, 2)
    # print('ops = {}' .format(ops))
    # print('pick_point = {}' .format(pick_point))
    # print('K_0 = {}' .format(K_0))
    # print('C_0 = {}' .format(C_0))
    flag, rvec, tvec = cv2.solvePnP(ops, pick_point, K_0, C_0, flags = cv2.SOLVEPNP_P3P)
    print('rvec = {}' .format(rvec))
    print('tvec = {}' .format(tvec))
    cv2.destroyWindow(info["pick_winname"])
    cv2.destroyWindow(info["zoom_winname"])
    return flag, rvec, tvec

def locate_record(camera_type, reverse, save = False, rvec = None, tvec = None):
    '''
    直接读取已储存的位姿，基于雷达每次位置变化不大
    这个函数也用来存储位姿

    :param camera_type:相机编号
    :param reverse:reverse r b 
    :param save:读取还是存储
    :param rvec:当存储时，将旋转向量填入
    :param tvec:当存储时，将平移向量填入

    :return: （当为读取模型时有用）读取成功标志，旋转向量，平移向量
    '''
    
    # 4.18 似乎没有用到相机编号

    max_order = -1 # 位姿文件个数
    max_file = None
    flag = False
    # 计算已存储的位姿文件中最大序号
    if not os.path.exists(LOCATION_SAVE_DIR):
        os.mkdir(LOCATION_SAVE_DIR)
    for f in os.listdir(LOCATION_SAVE_DIR):
        order, camera_num, f_enemy, _ = f.split('__')   # 文件里有序号信息，见save部分
        camera_num = int(camera_num)
        order = int(order)
        f_enemy = int(f_enemy)
        # 查询指定相机和敌方编号，一直查找直到找到最新的
        if f_enemy == reverse and camera_num == camera_type:
            if order > max_order:
                max_order = order
                max_file = f
    if save:
        # 序号__相机编号__红蓝方__时间
        filename = "{0}__{1}__{2}__{3}.txt".format(max_order+1, camera_type, reverse, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        with open(os.path.join(LOCATION_SAVE_DIR,filename),'w') as _log_f:
            _log_f.write("# rvec\n")
            _log_f.write(f"{float(rvec[0]):0.5f} {float(rvec[1]):0.5f} {float(rvec[2]):0.5f}\n")
            _log_f.write("# tvec\n")
            _log_f.write(f"{float(tvec[0]):0.5f} {float(tvec[1]):0.5f} {float(tvec[2]):0.5f}\n")
    elif max_order > -1:
        # 读取模型，若文件不为空
        flag = True
        pose = np.loadtxt(os.path.join(LOCATION_SAVE_DIR,max_file),delimiter=' ').reshape(2,3)
        rvec = pose[0]
        tvec = pose[1]

    return flag, rvec, tvec